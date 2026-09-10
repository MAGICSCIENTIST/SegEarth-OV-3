1. 单独编写 prompt 聚合函数

为了后面方便测试不同融合策略，可以写成：

def aggregate_prompt_scores(
    self,
    prompt_scores: torch.Tensor,
    mode: str = 'max',
    prompt_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Args:
        prompt_scores:
            [Q, H, W]，每个 prompt 的像素分数。

        mode:
            max、mean 或 weighted。

        prompt_weights:
            [Q]，每个 prompt 的可靠性权重。

    Returns:
        class_scores:
            [C, H, W]
    """
    class_scores = []

    for class_id in range(self.num_cls):
        mask = self.query_idx == class_id
        current_scores = prompt_scores[mask]  # [Qc, H, W]

        if current_scores.shape[0] == 0:
            score = torch.zeros_like(prompt_scores[0])

        elif mode == 'max':
            score = current_scores.max(dim=0).values

        elif mode == 'mean':
            score = current_scores.mean(dim=0)

        elif mode == 'weighted':
            if prompt_weights is None:
                raise ValueError(
                    'prompt_weights is required for weighted fusion.'
                )

            current_weights = prompt_weights[mask]
            current_weights = torch.softmax(
                current_weights,
                dim=0
            )

            score = (
                current_scores
                * current_weights[:, None, None]
            ).sum(dim=0)

        else:
            raise ValueError(
                f'Unsupported prompt fusion mode: {mode}'
            )

        class_scores.append(score)

    return torch.stack(class_scores, dim=0)

然后在 predict() 中：

prompt_scores = self._inference_single_view(image)

class_scores = self.aggregate_prompt_scores(
    prompt_scores,
    mode='max'
)

后续实验只需要改：

mode='max'
mode='mean'
mode='weighted'

不必反复改 predict() 主流程。


2. 注意：这里的 seg_logits 严格来说并不是原始 logits

这个实现中的变量名有些误导。

SAM 3 Processor 内部已经进行了：

out_masks = interpolate(...).sigmoid()

out_semantic_masks = interpolate(
    outputs["semantic_seg"],
    ...
).sigmoid()

presence_score = outputs[
    "presence_logit_dec"
].sigmoid()

因此：

masks_logits
semantic_mask_logits
presence_score

实际上已经是经过 sigmoid 的分数，通常处于 [0,1]，不是未归一化的 raw logits。

所以更准确的命名是：

prompt_scores
class_scores

而不是：

prompt_logits
class_logits

对于后续融合，不建议再次直接执行：

torch.sigmoid(prompt_scores)

因为这会重复 sigmoid，进一步压缩分数范围。


3. 需要区分三种 score。

score	形状	含义
prompt_scores[q]	[H,W]	prompt 的像素级响应图
presence_score[q]	标量	当前图像是否存在该 prompt
object_score[q,k]	标量	prompt 下第 k 个实例的置信度