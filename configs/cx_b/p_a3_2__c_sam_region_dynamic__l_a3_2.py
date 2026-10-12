_base_ = './base.py'

# Prompt=A3-2；Concept=sam_region_dynamic；Loss=A3-2。方法超参数统一在本配置链中管理。
cx_b = dict(
    prompt_method='A3-2',
    concept_method='sam_region_dynamic',
    loss_method='A3-2',
    data_mode='online_sam',
    loss=dict(
        prompt=dict(method='A3-2'),
        concept=dict(method='A3-2')))
