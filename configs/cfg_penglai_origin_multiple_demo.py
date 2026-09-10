_base_ = './base_config.py'

from concept_utils import load_concepts

concept_path = './configs/cls_penglai_origin_multiple_demo.json'
concepts = load_concepts(concept_path)
concept_classes = [
    concept.get('concept_name') or concept.get('concept_id')
    for concept in concepts
]
concept_palette = [
    [(37 * (idx + 1)) % 256, (97 * (idx + 3)) % 256,
     (53 * (idx + 5)) % 256]
    for idx in range(len(concepts))
]

# model settings
model = dict(
    classname_path=concept_path,
    prob_thd=0.1,
    bg_idx=255,
    prompt_aggregation='max',
    confidence_threshold=0.1,
    use_presence_score=True,
    use_transformer_decoder=True,
    slide_stride=512,
    slide_crop=512,
)

# dataset settings
dataset_type = 'YZTOriginMultipleDataset'
data_root = r'D:\learn\MyWork\mylib\output\temp\splitFile_raster_merge_g_dl2_origin_512_o02_filter_demo'

test_evaluator = dict(
    type='TNLSConceptIoUMetric',
    concept_path=concept_path,
    ignore_index=255)

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
        reduce_zero_label=False,
        metainfo=dict(classes=concept_classes, palette=concept_palette),
        data_prefix=dict(
            img_path='splitFile_raster_raw_all',
            seg_map_path='splitFile_dll3/all'),
        pipeline=test_pipeline))
