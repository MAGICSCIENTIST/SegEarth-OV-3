_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_loveda.txt',
    confidence_threshold=0.5,
    prob_thd=0.5,
)

# Preserve the original multiclass IoUMetric and add concept-preaggregation
# metrics for every individual text prompt.  With reduce_zero_label=True,
# LoveDA GT classes are zero based (background=0, ..., agricultural=6).
test_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore']),
    dict(
        type='PromptIoUMetric',
        concept_path='./configs/cls_loveda.txt',
        score_threshold=0.5,
        ignore_index=255),
]

# dataset settings
# Category labels: background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7. 
# And the no-data regions were assigned 0 which should be ignored. 
dataset_type = 'LoveDADataset'
# data_root = 'data/LoveDA'
data_root = r'E:\data\loveDa'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
