_base_ = './base.py'

# Prompt=dynamic_head；Concept=sam_region_dynamic；Loss=bce_tversky。方法超参数统一在本配置链中管理。
cx_b = dict(
    prompt_method='dynamic_head',
    concept_method='sam_region_dynamic',
    loss_method='bce_tversky',
    data_mode='online_sam',
    loss=dict(
        prompt=dict(method='bce_tversky'),
        concept=dict(method='bce_tversky')))
