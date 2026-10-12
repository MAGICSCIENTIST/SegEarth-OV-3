_base_ = './base.py'

# Prompt=dynamic_head；Concept=logsumexp；Loss=soft_iou_bce。方法超参数统一在本配置链中管理。
cx_b = dict(
    prompt_method='dynamic_head',
    concept_method='logsumexp',
    loss_method='soft_iou_bce',
    data_mode='cache',
    loss=dict(
        prompt=dict(method='soft_iou_bce'),
        concept=dict(method='soft_iou_bce')))
