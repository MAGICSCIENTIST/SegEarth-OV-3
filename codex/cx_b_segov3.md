# CX-B 的 segov3 prompt 方法

在现有 `train_batch_prompt_concept_variants.py` 入口选择
`--prompt_method=segov3`（也兼容原有 `--prompt_methd`）。继续使用现有训练、
checkpoint、concept 聚合和评估流程，没有独立的复原配置或训练入口。

新增 21 个普通组合配置：1 个 prompt 方法 × 7 个 concept 方法 × 3 个 loss。
全部方法现在共有 63 种组合。

## 评分与训练

`segov3` 固定使用原始 Eval LoveDA 公式：

```text
I = max_k(SAM mask_k × object_score_k)
prompt = max(semantic_score, I) × presence_score
```

它没有可训练的 prompt 参数，记录为 `status=fixed, best_epoch=0`。
`prompt_init=scratch` 表示直接使用该规则；不接受 A3 warmstart/reuse。
concept 仍可选择全部现有方法，可训练 concept 使用所选 loss 优化。
对于 `segov3 + max` 或 `segov3 + A3-2`（mean），两阶段都没有可训练参数，
三个 loss 不影响预测，也不生成用不到的训练集缓存。

## 复原 Eval LoveDA 的组合

```powershell
python train_batch_prompt_concept_variants.py --prompt_method=segov3 --concept_method=max --loss_method=A3-2 --prompt-init=scratch --output-root=E:/output/SEGOV3/weights/stage_cx_b --seed=3407
```

这里的 `loss_method=A3-2` 只是组合标识；该组合无需优化。
需要训练 concept 时，把 `max` 换成 `dynamic` 等，并照常选择三个 loss。

所有 segov3 组合配置均设置原始 SAM confidence threshold=0.5、
最终分类 threshold=0.5、低分像素 background=0、关闭滑窗；
保留原始语义、实例、presence 分支。
最终 concept 取 `max` 才对应 Eval LoveDA 的 concept 聚合；`A3-2` 是均值。
不要传入改变这些设置的 alternate base 或 cfg override 后再声称复原基线。

独立缓存为 `weights/prompt_loveda_independent/prompt_head_cache_segov3_fp32`，
保存 float32 原始 heads。旧 A3 缓存的 confidence threshold=0.1，
不能用于这次复原；缓存加载会检查 SAM 设置。

训练完成后使用已有评估入口：

```powershell
python eval_cx_b.py --pth-dir E:/output/SEGOV3/weights/stage_cx_b/p_segov3__c_max__l_a3_2/seed_3407 --out out/segov3_max_eval --vis-concept --vis-prompt
```

`panel` 使用原始 max/presence 规则展示各 prompt 分支。
预测 PNG 和 NPZ 使用 0..6；原始 MMSeg 保存的 LoveDA PNG 使用 1..7，
逐像素比较前需将后者减 1。GT 原始 0 被忽略，1..7 映射到 0..6。

比较基线 mIoU 时看 `multiclass_argmax`，不要使用训练阶段的独立二分类 mIoU。
CX-B 独立二分类延续 `>=0.5`；原始 PromptIoUMetric 使用 `>0.5`，
恰好等于阈值的像素会有差异；这不影响最终多分类复原。

## 已执行的复原验证

2026-09-19：从 1669 张 LoveDA 验证图中等距选取 2522、3356、4190，
每张 1024×1024，检查全部 11 个 prompt、7 个 concept。

运行 `python -m tools.verify_cx_b_segov3 --max-images 3`：
真实 SAM 分别执行原始源代码的推理与 CX-B heads 提取，后者经过 float32
磁盘缓存往返后调用新增 prompt 方法和现有 max 聚合器。

| 图片 | prompt 最大误差 | concept 最大误差 | 与原始推理差异像素 | 与历史结果差异像素 |
| --- | ---: | ---: | ---: | ---: |
| 2522.png | 0 | 0 | 0 | 0 |
| 3356.png | 0 | 0 | 0 | 0 |
| 4190.png | 0 | 0 | 0 | 0 |

历史结果取自 `E:/output/SEGOV3/out/cfg_loveda/result`。
完整抽样报告在 `out/cx_b_segov3_verification/report.json`。
这证明抽样结果逐像素复原；尚未全量重算 1669 张的 47.41% mIoU。
需要全量核对可运行相同验证工具并设置 `--max-images=-1`。
