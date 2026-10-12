"""Compare the composable segov3/max path with original Eval LoveDA inference.

Run from the repository root: python -m tools.verify_cx_b_segov3 --max-images 3
Uses real SAM weights, independent forwards, and a float32 disk cache roundtrip.
Only this verification tool reads original methods through AST, avoiding the
MMSeg runner/constructor and a second copy of the same SAM backbone on the GPU.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cx_b.configuration import resolve_combination_config
from cx_b.modules import build_concept_aggregator, build_prompt_module
from eval_cx_b import classify_scores, predict_image


def original_methods():
    path = Path(__file__).resolve().parents[1] / 'segearthov3_segmentor.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'SegEarthOV3Segmentation')
    names = ['_inference_single_view', 'aggregate_prompt_scores', 'predict']
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(torch=torch, F=F, Image=Image, PixelData=SimpleNamespace)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-images', type=int, default=3,
                        help='Evenly spaced validation images; -1 checks all.')
    parser.add_argument('--out', default='out/cx_b_segov3_verification')
    parser.add_argument('--base-results', default='E:/output/SEGOV3/out/cfg_loveda/result')
    args = parser.parse_args(argv)
    if args.max_images == 0 or args.max_images < -1:
        parser.error('--max-images must be positive or -1')
    import train_learnable_params as shared
    import train_prompt_params_independent as cache_utils
    from concept_utils import load_concepts
    from cx_b.trainer import expected_prompt_records

    cfg, _ = resolve_combination_config('segov3', 'max', 'A3-2')
    concepts = load_concepts(cfg['model']['classname_path'])
    records = expected_prompt_records(concepts, cfg['model']['classname_path'], 'segov3')
    dataset = shared.build_dataset(shared.get_validation_dataset_cfg(cfg))
    count = len(dataset) if args.max_images == -1 else min(args.max_images, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f'Building SAM; validating {count}/{len(dataset)} images, '
          f'{len(records)} prompts, {len(concepts)} concepts', flush=True)
    model = shared.build_model(cfg['model']).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    original = SimpleNamespace(**{key: getattr(model, key) for key in (
        'processor', 'device', 'query_words', 'query_idx', 'num_queries', 'num_cls',
        'use_sem_seg', 'use_transformer_decoder', 'use_presence_score',
        'slide_crop', 'slide_stride', 'prob_thd', 'bg_idx')}, prompt_aggregation='max')
    for name, method in original_methods().items():
        setattr(original, name, method.__get__(original))
    prompts = {r['prompt_index']: build_prompt_module('segov3', {}) for r in records}
    selected = list(range(len(concepts)))
    aggregators = {i: build_concept_aggregator('max', sum(
        r['concept_index'] == i for r in records)) for i in selected}
    report = dict(config=cfg, image_indices=indices, dataset_images=len(dataset),
                  prompt_count=len(records), concept_count=len(concepts), images=[])
    for index in indices:
        data = dataset[index]['data_samples']
        path = Path(data.metainfo['img_path'])
        image = Image.open(path).convert('RGB')
        sample = SimpleNamespace(metainfo=dict(img_path=str(path), ori_shape=(image.height, image.width)))
        sample.set_data = lambda values: sample.__dict__.update(values)
        with torch.no_grad():
            original.predict(None, [sample])
            heads = model.extract_prompt_heads_batch(image, range(len(records)))
            # Exercise the same serialization/conversion used by CX-B caching.
            cache_path = output / 'last_heads.pth'
            torch.save([cache_utils.cache_heads(h, 'float32') for h in heads], cache_path)
            cached = [{k: v.to(model.device) for k, v in h.items()}
                      for h in torch.load(cache_path, map_location='cpu')]
            scores, prompt_scores = predict_image(records, selected, prompts, aggregators,
                                                  lambda r: cached[r['prompt_index']])
        prediction = classify_scores(scores, selected, cfg['model']['prob_thd'],
                                     cfg['cx_b']['prediction_background'])
        original_prediction = sample.pred_sem_seg.data[0].cpu().numpy()
        row = dict(image=str(path),
                   prompt_max_abs_error=float(np.abs(np.stack(list(prompt_scores.values())) -
                       sample.raw_prompt_scores.data.cpu().numpy()).max()),
                   concept_max_abs_error=float(np.abs(scores - sample.seg_logits.data.cpu().numpy()).max()),
                   prediction_mismatch_pixels=int(np.count_nonzero(prediction != original_prediction)),
                   pixels=int(prediction.size))
        baseline_path = Path(args.base_results) / path.name
        if baseline_path.is_file():
            # MMSeg IoUMetric exports reduce_zero_label=True predictions as 1..7.
            baseline = np.array(Image.open(baseline_path)).astype(np.int32) - 1
            if baseline.shape != prediction.shape:
                raise ValueError(f'Baseline shape mismatch: {baseline_path}')
            row['historical_baseline_mismatch_pixels'] = int(np.count_nonzero(prediction != baseline))
        Image.fromarray(prediction).save(output / path.name)
        report['images'].append(row)
        report['passed'] = all(r['prompt_max_abs_error'] == 0 and r['concept_max_abs_error'] == 0
                               and r['prediction_mismatch_pixels'] == 0
                               and r.get('historical_baseline_mismatch_pixels', 0) == 0
                               for r in report['images'])
        (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
