import os.path as osp
import runpy

_base_vars = runpy.run_path(
    osp.join(osp.dirname(__file__), 'cfg_penglai_origin_multiple_learnable.py'))
globals().update({
    key: value
    for key, value in _base_vars.items()
    if not key.startswith('__')
})

# concept 指标实验目标：
# 1. 不使用 CE，不追求整体像素级平均交叉熵。
# 2. 只用 soft IoU / recall floor 推动每个 concept 的可分性。
# 3. 这是探索性配置，可能更容易让权重到达上下界。
learnable_train_cfg = dict(learnable_train_cfg)
learnable_train_cfg.update(
    loss_mode='concept_metric',

    # 关闭 CE，让训练目标只来自 concept 级 soft 指标。
    ce_loss_weight=0.0,

    # soft IoU 是主要目标。
    soft_iou_loss_weight=1.0,

    # 更强地约束低召回 concept。
    recall_loss_weight=0.5,
    recall_target=0.35,

    # 最差 concept 的 loss 额外加权，减少强类掩盖弱类的可能性。
    worst_concept_loss_weight=1.0,

    # 关闭参数正则，允许每个 prompt 的独立参数更自由地寻找指标最优点。
    param_reg_weight=0.0,

    # 保持 False：不存在的 concept 若被误预测，也会被 soft IoU 约束。
    soft_iou_present_only=False,
)
