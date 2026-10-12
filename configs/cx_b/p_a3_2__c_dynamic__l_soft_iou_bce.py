_base_ = './base.py'

# Prompt=A3-2；Concept=dynamic；Loss=soft_iou_bce。方法超参数统一在本配置链中管理。
cx_b = dict(
    prompt_method='A3-2',
    concept_method='dynamic',
    loss_method='soft_iou_bce',
    data_mode='cache',
    loss=dict(
        prompt=dict(method='soft_iou_bce'),
        concept=dict(method='soft_iou_bce')))
