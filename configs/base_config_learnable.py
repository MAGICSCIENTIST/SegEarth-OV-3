# base config for the learnable SegEarthOV3 calibration model

model = dict(
    type='SegEarthOV3LearnableSegmentation',
    model_type='SAM3-learnable',
    fusion_mode='max',
    prompt_fusion_mode='max',
    presence_gain=1.0,
    presence_bias=0.5,
    prompt_threshold=0.0,
    concept_threshold=0.0,
    concept_weight=1.0,
    min_prompt_component_area=0.0,
    instance_filter_use_presence=True,
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='BlackTextSegLocalVisualizer', vis_backends=vis_backends, alpha=0.7, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))
