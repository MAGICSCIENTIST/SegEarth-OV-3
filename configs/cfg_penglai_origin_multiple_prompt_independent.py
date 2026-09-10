import os.path as osp
import runpy


_base_vars = runpy.run_path(
    osp.join(
        osp.dirname(__file__),
        'cfg_penglai_origin_multiple_learnable.py'))
globals().update({
    key: value
    for key, value in _base_vars.items()
    if not key.startswith('__')
})


# Stage 1: optimize every prompt against the binary GT of its own concept.
# Concept aggregation and concept-level parameters remain fixed in this stage.
prompt_independent_train_cfg = dict(
    # Maximum optimization epochs for each independent prompt.
    epochs_per_prompt=80,
    lr=0.003,
    # The historical A3-0 baseline keeps its original validation schedule.
    # Stage A3-1+ launch entries override validation to every epoch and disable
    # early stopping so all prompts receive the same optimization budget.
    patience=12,
    validate_every_epochs=5,
    max_train_samples=-1,
    max_val_samples=-1,
    metric_threshold=0.5,
    selection_metric='soft_iou',
    # Stage 1 selects checkpoints by soft IoU, so optimize the same objective.
    # BCE remains in the logs for calibration diagnosis but has no loss weight.
    bce_loss_weight=0.0,
    soft_iou_loss_weight=1.0,
    recall_loss_weight=0.1,
    recall_target=0.3,
    max_positive_weight=20.0,
    param_reg_weight=0.01,
    grad_clip_norm=1.0,
    presence_gain_max=8.0,
    prompt_weight_max=3.0,
    cache_dtype='float16',
    progress_interval=20,
    # This controls image output only; it never skips validation metrics.
    val_visualize_every_epochs=1,
    # full_panel: diagnostic grid; normal: GT/prediction raster overlays only.
    visual_type='full_panel',
    seed=3407,
)
