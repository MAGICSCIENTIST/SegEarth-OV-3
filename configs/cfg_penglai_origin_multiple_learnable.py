_base_ = './base_config_learnable.py'

from concept_utils import load_concepts

# concept_path = './configs/cls_penglai_origin_multiple_5.json'
concept_path = './configs/cls_penglai_origin_multiple_5_justformtp_test.json'
concepts = load_concepts(concept_path)
concept_classes = [
    concept.get('concept_name') or concept.get('concept_id')
    for concept in concepts
]
concept_palette = [
    [(37 * (idx + 1)) % 256, (97 * (idx + 3)) % 256,
     (53 * (idx + 5)) % 256]
    for idx in range(len(concepts))
]

model = dict(
    classname_path=concept_path,
    prob_thd=0.1,
    bg_idx=255,
    fusion_mode='weighted_sum',
    prompt_fusion_mode='weighted_sum',
    semantic_weight=0.5,
    presence_gain=1.0,
    presence_bias=0.5,
    prompt_weight=1.0,
    prompt_threshold=0.0,
    concept_threshold=0.0,
    concept_weight=1.0,
    min_prompt_component_area=0.0,
    instance_filter_use_presence=True,
    confidence_threshold=0.1,
    slide_stride=512,
    slide_crop=512,
)

# train_learnable_params.py 专用训练配置。
# launch.json 现在只需要传 config 和 save-dir；TensorBoard 默认写到
# save-dir/tensorboard，模型权重、验证日志和可视化仍写到 save-dir。
learnable_train_cfg = dict(
    # 训练轮数与学习率。当前只训练 calibration 参数，不训练 SAM 主干。
    epochs=160,
    lr=0.003,

    # 调试/加速用限制。-1 表示不限制训练或验证样本数量。
    max_iters=-1,
    max_val_samples=-1,

    # 控制台进度、验证频率和 checkpoint 频率。
    progress_interval=10,
    validate_every_epochs=10,
    save_every_epochs=10,

    # 参数正则和梯度裁剪，防止可学习权重过快跑到极端值。
    param_reg_weight=0.01,
    grad_clip_norm=1.0,

    # 每次 optimizer.step 后对可学习权重做裁剪。
    prompt_weight_max=3.0,
    presence_gain_max=8.0,

    # 原始训练目标：只使用像素级 concept cross entropy。
    loss_mode='ce',
    ce_loss_weight=1.0,
    soft_iou_loss_weight=0.0,
    recall_loss_weight=0.0,

    # recall_target 只在 recall_loss_weight > 0 时生效。
    recall_target=0.3,

    # worst_concept_loss_weight > 0 时会额外惩罚当前样本最差 concept。
    worst_concept_loss_weight=0.0,

    # False 表示 soft IoU 同时惩罚不存在类别的误预测概率。
    soft_iou_present_only=False,
    soft_loss_eps=1e-6,
)

dataset_type = 'YZTOriginMultipleDataset'
data_root = r'D:\learn\MyWork\mylib\output\temp\splitFile_raster_merge_g_dl2_origin_512_o02_filter'

test_evaluator = dict(
    type='TNLSConceptIoUMetric',
    concept_path=concept_path,
    ignore_index=255)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

validation_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        metainfo=dict(classes=concept_classes, palette=concept_palette),
        data_prefix=dict(
            img_path='splitFile_raster_raw/train',
            seg_map_path='splitFile_dll3/train'),
        pipeline=train_pipeline))

validation_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        metainfo=dict(classes=concept_classes, palette=concept_palette),
        data_prefix=dict(
            img_path='splitFile_raster_raw/val',
            seg_map_path='splitFile_dll3/val'),
        pipeline=validation_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        metainfo=dict(classes=concept_classes, palette=concept_palette),
        data_prefix=dict(
            img_path='splitFile_raster_raw_all',
            seg_map_path='splitFile_dll3/all'),
        pipeline=test_pipeline))
