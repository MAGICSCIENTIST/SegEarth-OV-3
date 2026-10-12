_base_ = '../cfg_loveda_origin_multiple_learnable.py'

# CX-B 默认先独立训练 prompt，再独立训练 concept。可追加联调阶段；SAM3 始终冻结。
cx_b = dict(
    prompt_method='A3-2',
    # 默认从统一三参数和零动态残差开始训练，不读取或补训旧 A3。
    # 仅在显式选择 a3_warmstart/a3_reuse 时才需要 --a3-checkpoint-root。
    prompt_init='scratch',
    concept_method='A3-2',
    loss_method='A3-2',
    concept_independent=True,
    # 此字段描述聚合器仍按 concept 构建；joint_mode 控制是否追加联合优化。
    joint_mode='independent',
    joint_train=dict(
        epochs=20, lr=0.001, patience=7, validate_every_epochs=1,
        checkpoint_interval_epochs=10, save_last_checkpoint=True,
        threshold=0.5, background=0, temperature=1.0,
        competition_weight=1.0, binary_weight=0.25,
        param_reg_weight=0.0),
    data_mode='cache',
    resume=True,
    # 默认仅保存完整 pth，不重复导出验证集 concept 概率 NPY。
    # 可在训练时显式启用，或训练完成后使用 export-only 模式按需生成。
    export_probabilities=False,
    metric_threshold=0.5,
    prompt_head_cache=(
        'weights/prompt_loveda_independent/prompt_head_cache_mtptest'),
    prompt_train=dict(
        epochs=80,
        lr=0.003,
        batch_size=13,
        patience=20,
        validate_every_epochs=2,
        # 训练管线每隔这些 epoch 保存一个可加载的 epoch_XXXX.pth。
        checkpoint_interval_epochs=10,
        save_last_checkpoint=True,
        selection_metric='hard_iou',
        # 默认严格选择 epoch 0..N 中验证 IoU 最高的权重。
        identity_fallback=False,
        identity_fallback_min_delta=0.005,
        param_reg_weight=0.01,
        hidden_channels=16),
    concept_train=dict(
        epochs=80,
        lr=0.003,
        patience=20,
        validate_every_epochs=2,
        checkpoint_interval_epochs=10,
        save_last_checkpoint=True,
        selection_metric='hard_iou',
        param_reg_weight=0.0,
        hidden_channels=16,
        logsumexp_temperature=0.1),
    loss=dict(
        prompt=dict(
            soft_iou_weight=1.0,
            recall_weight=0.1,
            recall_target=0.3,
            bce_weight=1.0,
            positive_weight=1.0,
            tversky_weight=1.0,
            tversky_alpha_fp=0.6,
            tversky_beta_fn=0.4),
        concept=dict(
            soft_iou_weight=1.0,
            recall_weight=0.1,
            recall_target=0.3,
            bce_weight=1.0,
            positive_weight=1.0,
            tversky_weight=1.0,
            tversky_alpha_fp=0.6,
            tversky_beta_fn=0.4)),
    sam_region_cfg=dict(
        read_output_cache=False,
        write_output_cache=False,
        amp_dtype='bfloat16',
        quality_source='object_score',
        min_region_score=0.0,
        max_masks_per_prompt=64,
        mask_threshold=0.5,
        mask_processing_block_size=16))
