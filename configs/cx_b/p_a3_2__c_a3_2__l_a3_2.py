_base_ = './base.py'

# Prompt=A3-2；Concept=A3-2；Loss=A3-2。方法超参数统一在本配置链中管理。
cx_b = dict(
    prompt_method='A3-2',
    concept_method='A3-2',
    loss_method='A3-2',
    data_mode='cache',
    loss=dict(
        prompt=dict(method='A3-2'),
        concept=dict(method='A3-2')))
