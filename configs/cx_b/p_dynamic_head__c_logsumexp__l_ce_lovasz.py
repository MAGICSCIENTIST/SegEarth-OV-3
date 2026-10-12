_base_ = './p_dynamic_head__c_logsumexp__l_a3_2.py'

# 独立 prompt/concept 阶段沿用 A3-2；多分类联调增加 Lovasz-Softmax。
# 保留原组合配置，输出通过 l_ce_lovasz 与旧实验分开。
cx_b = dict(
    loss_method='ce_lovasz',
    joint_mode='multiclass_joint',
    joint_train=dict(loss_method='ce_lovasz', lovasz_weight=0.25))
