import os.path as osp
import runpy

_base_vars = runpy.run_path(
    osp.join(osp.dirname(__file__), 'cfg_penglai_origin_multiple_learnable.py'))
globals().update({
    key: value
    for key, value in _base_vars.items()
    if not key.startswith('__')
})

# 混合训练目标：
# 1. CE 保留像素级监督，避免训练过度偏离原始分割竞争关系。
# 2. soft IoU 让每个 concept 都进入可微优化目标。
# 3. recall floor 针对低召回 concept 增加惩罚。
learnable_train_cfg = dict(learnable_train_cfg)
learnable_train_cfg.update(
    loss_mode='mixed',

    # CE 仍作为主损失，提供稳定梯度。
    ce_loss_weight=1.0,

    # per-concept soft IoU 辅助优化，重点改善弱类 IoU。
    soft_iou_loss_weight=0.3,

    # 低于 recall_target 的 concept 会被额外惩罚。
    recall_loss_weight=0.1,
    recall_target=0.3,

    # 当前样本中最差 concept 额外加权，避免只优化平均值。
    worst_concept_loss_weight=0.3,

    # 保持 False：不存在的 concept 若被误预测，也会被 soft IoU 约束。
    soft_iou_present_only=False,
)
