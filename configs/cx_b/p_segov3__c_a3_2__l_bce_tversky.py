_base_ = './base.py'

# Original Eval LoveDA prompt rule, composable with every concept/loss.
# Raw heads use the same SAM filtering and retain float32 cache precision.
model = dict(
    confidence_threshold=0.5,
    prob_thd=0.5,
    bg_idx=0,
    use_sem_seg=True,
    use_transformer_decoder=True,
    use_presence_score=True,
    instance_filter_use_presence=True,
    slide_stride=0,
    slide_crop=0,
)

cx_b = dict(
    prompt_method='segov3',
    concept_method='A3-2',
    loss_method='bce_tversky',
    data_mode='cache',
    cache_dtype='float32',
    prompt_head_cache=(
        'weights/prompt_loveda_independent/prompt_head_cache_segov3_fp32'),
    prediction_background=0,
    loss=dict(
        prompt=dict(method='bce_tversky'),
        concept=dict(method='bce_tversky')))

