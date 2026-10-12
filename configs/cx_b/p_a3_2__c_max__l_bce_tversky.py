_base_ = './base.py'

# Prompt=A3-2；Concept=max；Loss=bce_tversky。方法超参数统一在本配置链中管理。
cx_b = dict(
    prompt_method='A3-2',
    concept_method='max',
    loss_method='bce_tversky',
    data_mode='cache',
    loss=dict(
        prompt=dict(method='bce_tversky'),
        concept=dict(method='bce_tversky')))
