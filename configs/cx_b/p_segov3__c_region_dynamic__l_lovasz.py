_base_ = './p_segov3__c_region_dynamic__l_a3_2.py'

# Independent prompt/concept warmup keeps A3-2; multiclass training uses pure Lovasz.
cx_b = dict(
    loss_method='lovasz',
    joint_mode='multiclass_joint',
    joint_train=dict(loss_method='lovasz', competition_weight=0.0,
                     binary_weight=0.0, lovasz_weight=1.0))
