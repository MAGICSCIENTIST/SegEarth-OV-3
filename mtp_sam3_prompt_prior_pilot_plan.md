# MTP 接入 SAM3 Prompt 连续概率图的小型验证方案

## 1. 目标

在不重新运行 SAM3、也不先改动现有 MTP 主模型的前提下，使用 SegEarth-OV-3 已缓存的 prompt head 和 A3-2-mtptest 最优参数，导出少量 prompt 的连续 `final_score` 概率图，并验证这些多通道先验是否能改善 MTP 的七类语义分割结果。

目标 MTP 启动配置：

`D:\learn\MTP\.vscode\launch.json` 中的：

`finetune_yzt_g_dl2_origin_merge_penglai-filter.py`

本方案首先进行低成本、可归因的离线 logit fusion probe。只有 probe 稳定有效，才进入 MTP backbone 的多通道联合训练。

## 2. 可行性结论

结论：技术上可行，适合先做小型探索实验，但当前结果不能直接作为无泄漏的正式对比结论。

支持可行性的依据：

1. SAM3 cache 与 MTP 数据使用完全一致的 train/val 图像 ID：train 411 张，val 112 张，文件名集合一致。
2. 两边图像都是 `512 x 512`，可以按图像 stem 对齐概率图。
3. SAM3 使用 `splitFile_raster_raw`，MTP 使用 `splitFile_raster`。两者像素值不同，但抽样中 MTP 的一个来源通道与 raw 图像通道逐像素一致，表明它们处于同一空间网格。
4. SAM3 cache 已包含每张图、每个 prompt 的 `instance_score_map`、`semantic_score_map` 和 `presence_score`，无需重新运行 SAM3。
5. A3-2-mtptest 已输出每个 prompt 的 selected 参数，可以从 head 精确重建连续 `final_score`。
6. MTP 当前最佳验证 checkpoint 是 `iter_60000.pth`，val mIoU 为 `75.27`，可以作为固定基线。

当前限制：

1. A3-2-mtptest 的 prompt 参数使用了当前 112 张 val 做 best epoch 选择。
2. MTP 的 `iter_60000.pth` 也是根据同一 val 上的结果选出的较优 checkpoint。
3. 因此首轮 probe 只能判断“是否值得继续”，不能作为无偏泛化结论或论文最终结果。
4. 正式实验需要新的 held-out test，或者重新划分 prompt 优化的 train/dev/test。

## 3. 已核对的源数据

### 3.1 SAM3 head cache

```text
D:\learn\SegEarth-OV-3\weights\prompt_independent\prompt_head_cache_mtptest
```

缓存签名：

```text
confidence_threshold = 0.1
use_sem_seg = true
use_transformer_decoder = true
instance_filter_use_presence = true
cache_dtype = float16
```

缓存规模：

```text
train: 18 prompts x 411 images
val:   18 prompts x 112 images
total: 9414 pth files, about 9.89 GB
```

### 3.2 Prompt 参数结果

```text
E:\output\SEGOV3\weights\stage_a\A3-2-mtptest\seed_3407\prompt_optimization_results.json
```

导出时必须按 `prompt_index + concept_id` 取记录，不能只按 prompt 字符串取记录，因为索引 9 和索引 17 都叫 `bare soil`，但属于不同 concept。

### 3.3 MTP 数据与基线

MTP 数据根目录：

```text
D:\learn\MyWork\mylib\output\temp\splitFile_raster_merge_g_dl2_origin_512_o02_filter
```

MTP 输入：

```text
splitFile_raster\train
splitFile_raster\val
```

SAM3 输入：

```text
splitFile_raster_raw\train
splitFile_raster_raw\val
```

MTP coarse GT：

```text
splitFile_current\train
splitFile_current\val
```

SAM3 prompt 优化使用的 fine GT：

```text
splitFile_dll3\train
splitFile_dll3\val
```

MTP 基线 checkpoint：

```text
D:\learn\MTP\output\multitask_pretrain\finetune\Semantic_Segmentation\yzt\rvsa-l-upernet-512-mae-mtp-lyzt-g-dl2-origin-merge-penglai-filter\iter_60000.pth
```

该 checkpoint 的 val 指标：

| Class | IoU |
|---|---:|
| Farmland | 84.15 |
| Forest | 86.17 |
| Grass | 63.79 |
| Wetland | NaN，当前 val 无该类 |
| Water | 86.68 |
| Residential | 93.54 |
| WasteLand | 37.27 |
| mIoU | 75.27 |

## 4. Fine concept 与 MTP coarse class 的实际对齐

以下结果由 112 张 val 的 fine/coarse GT 逐像素统计得到。

| SAM3 concept | 主要对应 MTP class | concept 像素纯度 | MTP class 覆盖率 | 二值 IoU | 结论 |
|---|---|---:|---:|---:|---|
| agricultural_facility | Residential | 90.07% | 66.47% | 61.93% | 可作为 Residential 子类先验 |
| cultivated_field | Farmland | 98.33% | 85.09% | 83.87% | 对齐很好 |
| forestAndShrubland | Forest | 89.23% | 99.19% | 88.58% | 对齐很好 |
| pond_or_reservoir | Water | 99.25% | 99.31% | 98.58% | 与 Water 几乎完整对齐 |
| river_or_canal | Water | 98.32% | 7.36% | 7.35% | 很纯，但仅覆盖 Water 小子集 |
| bare_or_excavated_surface | 无单一 coarse class | WasteLand 仅占该 concept 6.15% | 覆盖 WasteLand 44.26% | 5.70% | 不能直接映射为 WasteLand |

`bare_or_excavated_surface` 的 concept 像素实际分布约为：Farmland 49.90%、Forest 25.68%、Grass 12.99%、WasteLand 6.15%、Residential 5.16%。因此 `bare soil` 只能作为探索性通用特征，不能当作 WasteLand 概率。

## 5. 建议导出的 prompt 通道

### 5.1 P5-aligned 主实验

导出以下 5 个独立连续通道：

| Channel | prompt_index | Prompt | Concept | Selected hard IoU | 预期用途 |
|---:|---:|---|---|---:|---|
| 0 | 1 | building | agricultural_facility | 0.3361 | Residential 子类先验 |
| 1 | 4 | lake | pond_or_reservoir | 0.1197 | Water 面状子类先验 |
| 2 | 5 | river | river_or_canal | 0.1806 | Water 线状子类先验 |
| 3 | 11 | plowed field | cultivated_field | 0.2001 | Farmland 纹理先验 |
| 4 | 15 | scattered trees | forestAndShrubland | 0.3661 | Forest 先验 |

这里保留 `building` 的 identity fallback 参数。identity fallback 表示训练未改善它，不表示该 prompt 不可用；它的初始化结果反而是农业设施组中最强的。

### 5.2 P6-with-bare 探索实验

在 P5 基础上增加：

| Channel | prompt_index | Prompt | Concept | Selected hard IoU | 注意 |
|---:|---:|---|---|---:|---|
| 5 | 17 | bare soil | bare_or_excavated_surface | 0.2865 | concept 与 MTP coarse class 混合，仅允许自由学习，不做 WasteLand 硬映射 |

P5 与 P6 必须作为两个独立 launch 实验。P6 只有在优于 P5 且 shuffled control 无增益时，才能认为 `bare soil` 提供了有效补充信息。

## 6. Final score 重建规则

对每张图、每个选定 prompt，从对应 cache `.pth` 读取：

```text
instance_score_map
semantic_score_map
presence_score
```

从 `prompt_optimization_results.json` 的该 prompt selected `parameters` 读取：

```text
semantic_weight
presence_gain
presence_bias
prompt_weight
```

本次实验的 `fusion_mode` 是 `weighted_sum`，连续概率图必须按以下公式重建：

```python
eps = 1e-6
fused = (
    semantic_weight * semantic_score_map
    + (1.0 - semantic_weight) * instance_score_map
)
fused = fused.clamp(eps, 1.0 - eps)
final_logit = (
    torch.logit(fused)
    + presence_gain * (presence_score - presence_bias)
    + math.log(prompt_weight)
)
final_score = torch.sigmoid(final_logit)
```

要求：

1. 不使用 `metric_threshold=0.5` 做二值化。
2. 不输出最终 predict mask 作为输入特征。
3. 每个 prompt 保留一个独立 `float16` 连续通道。
4. 导出计算建议使用 float32，保存时再转 float16。
5. 输出前检查 shape、finite、范围 `[0, 1]`。

## 7. 推荐的最小实验：冻结 MTP 的离线 logit fusion probe

### 7.1 为什么先做该实验

不建议第一步就把 MTP patch embedding 从 3 通道改为 8/9 通道，因为：

1. 会改变预训练 patch embedding 权重形状。
2. 会同时引入数据增强、归一化、checkpoint 加载和优化器分组等变量。
3. 一旦 mIoU 变化，很难判断是 SAM3 信息有效，还是继续训练/初始化差异造成。
4. RVSA-L 完整训练开销较大，而一个 `K -> 7` 的 1x1 融合器只有几十个参数。

首轮使用冻结的 MTP logits：

```python
fused_logits = frozen_mtp_logits + adapter(normalized_sam3_prior)
```

其中 `adapter` 是零初始化、无 bias 的 `Conv2d(K, 7, kernel_size=1)`。epoch 0 时输出必须与 MTP baseline 完全一致。

### 7.2 需要在 MTP 新增的文件

建议新增，不修改已有训练入口行为：

```text
tools/export_sam3_prompt_priors.py
tools/export_mtp_logits.py
tools/train_sam3_prior_logit_probe.py
tools/evaluate_sam3_prior_logit_probe.py
```

可选测试文件：

```text
tests/test_sam3_prompt_prior_export.py
tests/test_sam3_prior_logit_fusion.py
```

### 7.3 `export_sam3_prompt_priors.py`

职责：

1. 读取 train/val manifest。
2. 按 prompt index 定位 cache 目录。
3. 按 `prompt_index + concept_id` 读取 selected 参数。
4. 重建每个 prompt 的 `final_score`。
5. 按原图 stem 输出一个 `.npz`。
6. 输出全局 `manifest.json`。

建议参数：

```text
--cache-dir
--results-json
--output-dir
--splits train val
--prompt-indices 1 4 5 11 15 17
--dtype float16
--overwrite
--verify-only
```

建议单图格式：

```python
np.savez_compressed(
    output_path,
    scores=scores.astype(np.float16),  # [K, H, W]
)
```

全局 manifest 至少记录：

```text
source cache path
source result JSON path
cache signature
split
image count
channel order
prompt index
prompt string
concept id
selected epoch
selected parameters
score formula/version
dtype
shape
per-channel min/max/mean/std
```

建议输出目录：

```text
E:\output\MTP\sam3_prompt_priors\A3-2-mtptest\seed_3407\train
E:\output\MTP\sam3_prompt_priors\A3-2-mtptest\seed_3407\val
```

### 7.4 `export_mtp_logits.py`

职责：

1. 加载现有 filter 配置和 `iter_60000.pth`。
2. 使用确定性的 512 val-style pipeline 分别遍历 train 与 val。
3. 调用模型 `predict`/`inference`，保存 postprocess 前的 7 通道 `seg_logits`。
4. 同时保存 coarse GT 或 GT 路径、图像 stem 和元数据。
5. 用 argmax 重算 baseline 指标。

注意：train split 导出时不能使用 `RandomResize`、`RandomCrop`、`RandomFlip` 或 `PhotoMetricDistortion`。

建议参数：

```text
config
checkpoint
--split train|val
--output-dir
--device cuda:0
--dtype float16
--verify-metrics
```

验收条件：

```text
val image count = 112
argmax mIoU 与 75.27 的差异 <= 0.05 个百分点
每类 IoU 与原日志基本一致
```

建议输出目录：

```text
E:\output\MTP\sam3_prompt_prior_probe\mtp_logits\iter_60000\train
E:\output\MTP\sam3_prompt_prior_probe\mtp_logits\iter_60000\val
```

### 7.5 `train_sam3_prior_logit_probe.py`

输入：

```text
frozen MTP logits: [7, H, W]
SAM3 final scores: [K, H, W]
coarse GT: [H, W]
```

建议先把 prior 转为稳定 logit 特征：

```python
prior = prior.clamp(1e-4, 1.0 - 1e-4)
prior = torch.logit(prior).clamp(-8.0, 8.0)
prior = (prior - train_channel_mean) / train_channel_std.clamp_min(1e-6)
```

融合器：

```python
adapter = nn.Conv2d(K, 7, kernel_size=1, bias=False)
nn.init.zeros_(adapter.weight)
fused_logits = mtp_logits.float() + adapter(prior.float())
```

训练建议：

```text
optimizer = AdamW
lr = 1e-3
weight_decay = 1e-4
epochs = 30
patience = 8
batch_size = 2 to 8
loss = CrossEntropyLoss
ignore_index = 255
class_weight = [0.45544791, 0.17755622, 1.78758481, 0, 1.52753874, 0.42757936, 1.61412342]
seeds = 3407, 1, 2
selection_metric = val mIoU
```

必须输出：

```text
baseline mIoU and per-class IoU
best fused mIoU and per-class IoU
delta mIoU and per-class delta
adapter weight matrix [7, K]
per-epoch train/val loss
per-epoch mIoU/mAcc/aAcc
TensorBoard
best checkpoint
args.json
channel manifest snapshot
log.txt
```

### 7.6 必须包含的对照

每个 P5/P6 实验均需有：

1. `baseline`：adapter 为零，不训练，必须精确复现冻结 MTP 指标。
2. `real_prior`：图像与 SAM3 prior 正确配对。
3. `shuffled_prior`：在 train 内固定打乱图像与 prior 的对应关系，val 也使用独立固定打乱。

只有 real prior 明显优于 baseline，且 shuffled prior 无类似增益，才说明空间先验确实有效。

## 8. 建议新增的 launch.json 配置

所有路径和实验参数尽量从 launch 传入，不写死在脚本中。

建议新增以下配置项：

```text
SAM3 Prior Pilot 1 - Export Priors P6
SAM3 Prior Pilot 2a - Export MTP Logits Train
SAM3 Prior Pilot 2b - Export MTP Logits Val
SAM3 Prior Pilot 3a - Probe P5 Aligned Seed 3407
SAM3 Prior Pilot 3b - Probe P6 With Bare Seed 3407
SAM3 Prior Pilot 3c - Probe P5 Shuffled Seed 3407
SAM3 Prior Pilot 3d - Probe P6 Shuffled Seed 3407
```

确认单 seed 有希望后，再把 `seed` 抽成 launch input，补跑 `1` 和 `2`。

建议 launch input：

```jsonc
{
  "id": "sam3PriorSeed",
  "type": "pickString",
  "description": "SAM3 prior probe seed",
  "options": ["3407", "1", "2"],
  "default": "3407"
}
```

P5 与 P6 的通道选择必须显式传入：

```text
P5: --prompt-indices 1 4 5 11 15
P6: --prompt-indices 1 4 5 11 15 17
```

## 9. Go/No-Go 标准

该 probe 的目标不是证明新方法，而是决定是否值得改 MTP 主模型。

建议标准：

### Go

满足全部条件：

1. 三个 seed 的平均 mIoU 相对 frozen baseline 提升至少 `+0.50` 个百分点。
2. 没有 seed 下降超过 `-0.20` 个百分点。
3. 至少一个目标 class 的 IoU 稳定提升 `+1.00` 个百分点。
4. shuffled prior 的平均增益小于 `+0.10` 个百分点。
5. adapter 权重不是全部接近 0，且不同 seed 的主要权重符号基本一致。

### Inconclusive

```text
平均 mIoU 增益在 +0.20 到 +0.50 之间
```

此时可以增加 spatial adapter，例如 `3x3 depthwise + 1x1`，但仍保持 MTP frozen。

### No-Go

满足任一条件：

```text
平均增益 < +0.20
real 与 shuffled 增益接近
只有单 seed 有提升
P6 的提升完全来自 bare soil 且跨 seed 不稳定
```

## 10. Probe 通过后的 MTP 多通道联合训练

只有完成第 7 至第 9 节并达到 Go，才实现本节。

### 10.1 新增数据 transform

建议新增：

```text
RS_Tasks_Finetune/Semantic_Segmentation/mmseg/datasets/transforms/loading_prompt_scores.py
```

注册 `LoadSAM3PromptScores`：

1. 按 `img_path` stem 加载 `.npz`。
2. 检查 `[K, H, W]` 与图像 H/W 一致。
3. 转为 `[H, W, K]` 并拼接到 `results['img']`。
4. 保存 channel manifest/version 到 metainfo。
5. 缺文件、shape 不一致、NaN 时立即报错，不允许静默补零。

为保证增强同步，train pipeline 顺序建议改为：

```text
LoadSingleRSImageFromFile
LoadAnnotationsTIF
PhotoMetricDistortion              # 仅 RGB，此时尚未拼 prior
LoadSAM3PromptScores               # 拼接连续 prior
RandomResize                       # RGB 与 prior 同步，连续通道使用图像插值
RandomCrop
RandomFlip
PackSegInputs
```

### 10.2 处理 MTP 现有 data preprocessor 的通道限制

当前 `SegDataPreProcessor` 只有输入恰好 3 通道时才执行 `bgr_to_rgb`；输入变成 `3 + K` 后不会换序，而且 3 元 mean/std 也无法广播到多通道输入。

必须选择一种明确方案：

推荐方案：在 `LoadSAM3PromptScores` 拼接前显式交换前三个 RGB/BGR 通道，然后配置：

```python
bgr_to_rgb = False
mean = original_rgb_mean + [0.0] * K
std = original_rgb_std + [1.0] * K
```

或者新增 `PromptSegDataPreProcessor`，仅对前三个通道做颜色转换和原有归一化，对 prior 做单独归一化。后者更清晰，但修改面略大。

### 10.3 新增零初始化输入适配器

不要直接把 RVSA patch embedding 改成 `3 + K` 输入并随机初始化。

建议新增一个 backbone 类型，例如：

```text
RVSA_MTPPromptFusion
```

逻辑：

```python
rgb = x[:, :3]
prior = x[:, 3:]
rgb = rgb + prompt_adapter(prior)
features = original_rvsa(rgb)
```

其中：

```python
prompt_adapter = nn.Conv2d(K, 3, kernel_size=1, bias=True)
nn.init.zeros_(prompt_adapter.weight)
nn.init.zeros_(prompt_adapter.bias)
```

这样加载 `iter_60000.pth` 后，训练前输出应与原模型一致，同时原始 3 通道 patch embedding 权重无需改变。

### 10.4 新增独立 config，不覆盖原 config

建议复制为：

```text
RS_Tasks_Finetune/Semantic_Segmentation/configs/mtp/yzt/rvsa-l-upernet-512-mae-mtp-lyzt-g-dl2-origin-merge-penglai-filter-sam3-prior.py
```

要求：

1. `load_from` 指向 `iter_60000.pth`。
2. backbone 内原 `pretrained` 设为 `None`，避免重复加载 encoder。
3. 第一轮只训练 prompt adapter，冻结原 backbone 与 UPerHead。
4. 第二轮仅在第一轮有效时，解冻 UPerHead 和最后若干 RVSA block，并使用更小 LR。
5. 必须增加一个无 prior、相同继续训练步数的 paired continuation control。

### 10.5 联合训练前的强制等价测试

在 adapter 全零时，对同一 val 图像比较原模型与新模型：

```text
max_abs_logit_diff <= 1e-5
argmax mask 完全一致
val mIoU 差异 <= 0.01 个百分点
```

未通过等价测试时，不允许开始训练，否则无法把后续变化归因于 SAM3 prior。

## 11. 实施检查清单

- [ ] 不修改原有 filter config 和原 launch 项。
- [ ] 新脚本参数均可从 launch 传入。
- [ ] cache prompt 必须按 index 校验，特别是两个 `bare soil`。
- [ ] 导出的是连续 final score，不做 0.5 threshold。
- [ ] train/val 各自严格按 stem 对齐，缺一张即失败。
- [ ] priors、MTP logits、GT 的 shape 完全一致。
- [ ] step 0 adapter 能精确复现 `iter_60000` baseline。
- [ ] 同时运行 real prior 与 shuffled prior。
- [ ] 输出 per-class IoU，不能只看 mIoU。
- [ ] P5 与 P6 分开保存结果。
- [ ] 记录 seed、args、channel manifest 和源文件 hash。
- [ ] 首轮结果明确标记为 exploratory validation reuse。
- [ ] 正式结论使用新的 held-out test 或重新划分数据。

## 12. 预期判断

从现有指标看，SAM3 prior 在技术上可以接入，但获得较大总体 mIoU 增益的概率不算高：

1. MTP 的 Farmland、Forest、Water、Residential 已达到 84 至 94 IoU，而较可靠的 SAM3 prompt 主要也对应这些强类。
2. MTP 最弱的 WasteLand 没有一个与 coarse class 干净对齐的 prompt。
3. `grassland` 和 `grass` 当前 prompt 几乎无有效响应，无法直接帮助 Grass。
4. SAM3 的价值更可能体现在局部边界、子类召回和少数难例，而不是直接替代 MTP 分类结果。

因此建议先执行冻结 MTP 的 logit fusion probe。它成本低、结果容易解释，并能在改动 RVSA 主干前回答最核心的问题：这些连续 prompt 概率图是否包含 MTP logits 尚未利用的互补信息。
