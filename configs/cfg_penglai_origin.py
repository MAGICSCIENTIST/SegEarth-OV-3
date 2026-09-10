_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_penglai_origin.txt',
    prob_thd=0.1,
    confidence_threshold=0.1,
    slide_stride=512,
    slide_crop=512,
)

# dataset settings
dataset_type = 'YZTOriginDataset'
data_root = r'D:\learn\MyWork\mylib\output\temp\splitFile_raster_merge_g_dl2_origin_512_o02_filter'

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
            img_path='splitFile_raster_raw/val',
            seg_map_path='splitFile_current/val'),
        pipeline=test_pipeline))
