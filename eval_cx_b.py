"""Evaluate saved CX-B prompt/concept checkpoints and export raster results.

Example:
    python eval_cx_b.py --pth-dir weights/stage_cx_b/<combination>/seed_3407 \
        --out out/cx_b_eval --vis-concept --vis-prompt

The saved run configuration is authoritative. Cached runs require their frozen
head cache; online SAM runs reconstruct heads using the saved model settings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cx_b.modules import SegOV3PromptAdapter, load_variant
from cx_b.configuration import refiner_settings
from cx_b.refiners import ConceptModules, JOINT_FORMATS, validate_joint_refiner, load_refiner
from cx_b.refiners import evaluation_refiner_options, ablation_folder
from cx_b.reporting import prediction_changes
from tools.visual_utils import (
    concept_color, render_labeled_overlay, safe_filename, save_binary_comparison,
    save_labeled_comparison, save_panel,
)

REPO_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
NODATA = 65535


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pth-dir', '--checkpoint-dir', required=True,
                        help='CX-B seed directory containing run_config.json.')
    parser.add_argument('--out', '--output-dir', required=True)
    parser.add_argument('--vis-concept', action=argparse.BooleanOptionalAction,
                        default=False, help='Save per-image concept overlays.')
    parser.add_argument('--vis-prompt', action=argparse.BooleanOptionalAction,
                        default=False, help='Save per-image prompt overlays.')
    parser.add_argument('--split', choices=['val', 'train', 'test'], default='val')
    parser.add_argument('--data-root', help='Override the saved dataset root.')
    parser.add_argument('--image-dir', help='Override the split image directory.')
    parser.add_argument('--gt-dir', help='Override the split annotation directory.')
    parser.add_argument('--prompt-head-cache', help='Override the raw head cache root.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Concept/prompt binary threshold (saved default: 0.5).')
    parser.add_argument('--prediction-threshold', type=float, default=None,
                        help='Reject max concept scores below this value; defaults '
                             'to saved model.prob_thd, as in eval.py.')
    parser.add_argument('--alpha', type=float, default=0.55)
    parser.add_argument('--joint-refiner', choices=('none', 'ov_class_spatial'), default=None,
                        help='Explicit refiner ablation; default follows the saved run.')
    parser.add_argument('--joint-refiner-strength', type=float, default=None)
    parser.add_argument('--max-images', type=int, default=-1,
                        help='Limit images for a smoke run; -1 evaluates all.')
    args = parser.parse_args(argv)
    for name in ['threshold', 'prediction_threshold', 'alpha']:
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            parser.error(f'--{name.replace("_", "-")} must be between 0 and 1.')
    if args.max_images == 0 or args.max_images < -1:
        parser.error('--max-images must be positive or -1.')
    return args


def resolve_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_run(directory, device, checkpoint_overrides=None, joint_checkpoint=None,
             joint_refiner=None, refiner_strength=None):
    """Load best modules (or explicit replacements), validating every identity.

    Overrides map run-relative best.pth paths to run-relative sibling variants.
    This never changes the training files or the saved run configuration.
    """
    directory = Path(directory).expanduser().resolve()
    run = json.loads((directory / 'run_config.json').read_text(encoding='utf-8'))
    config = run['config']
    # Snapshot content prevents edits to today's concept file changing old runs.
    content = run.get('inputs', {}).get('concept_content')
    concept_path = config.get('concept_path', config['model'].get('classname_path', ''))
    is_json = str(concept_path).lower().endswith('.json')
    if content is not None:
        if is_json:
            concepts = [c for group in json.loads(content).get('groups', [])
                        for c in group.get('concepts', [])]
        else:
            prompts = [[p.strip() for p in line.split(',') if p.strip()]
                       for line in content.splitlines() if line.strip()]
            concepts = [dict(concept_id=p[0], concept_name=p[0], prompts=p)
                        for p in prompts]
    else:
        concepts = config.get('concepts')
    if not concepts:
        raise ValueError('run_config.json does not contain concept definitions.')
    selected = run.get('inputs', {}).get('selected_concepts')
    selected = list(range(len(concepts))) if selected is None else sorted(set(selected))
    if not selected or any(i < 0 or i >= len(concepts) for i in selected):
        raise ValueError('Invalid/empty selected_concepts in run snapshot.')
    cx = config['cx_b']
    records, prompts, aggregators = [], {}, ConceptModules()
    prompt_folder = 'dynamic_prompt' if cx['prompt_method'] == 'dynamic_head' else 'prompt_checkpoints'
    fingerprint = run.get('fingerprint')
    overrides = dict(checkpoint_overrides or {})
    consumed = set()
    bundle = None
    saved_refiner, _ = refiner_settings(cx)
    if saved_refiner != 'none' and joint_checkpoint is None:
        joint_checkpoint = 'joint/best.pth'
    if joint_checkpoint is not None:
        if overrides:
            raise ValueError('Choose a whole joint checkpoint or module replacements, not both')
        bundle = torch.load(directory/joint_checkpoint, map_location=device)
        if bundle.get('format') not in JOINT_FORMATS or bundle.get('run_fingerprint') != fingerprint:
            raise ValueError(f'Joint checkpoint method/run mismatch: {joint_checkpoint}')
        validate_joint_refiner(bundle)
        if saved_refiner != 'none' and bundle['format'] != 'cx_b_joint_v2':
            raise ValueError('Saved R1 run is missing its joint v2 refiner')
        if bundle['format'] == 'cx_b_joint_v2' and selected != list(range(len(concepts))):
            raise ValueError('Evaluate all query concepts before selecting R1 output channels')
    bundle_consumed = set()

    def checked_load(path, kind, method):
        key = path.relative_to(directory).as_posix()
        if key in overrides:
            replacement = (directory / overrides[key]).resolve()
            if replacement.parent != path.parent.resolve() or replacement.suffix != '.pth':
                raise ValueError(f'Checkpoint replacement must be a sibling pth: {replacement}')
            consumed.add(key)
            path = replacement
        payload = None
        if bundle is not None:
            if key not in bundle['modules']:
                raise ValueError(f'Joint checkpoint is missing module: {key}')
            payload = bundle['modules'][key]
            if (payload.get('metadata', {}).get('joint_epoch') != bundle.get('epoch')
                    or payload.get('metadata', {}).get('joint_mode') != bundle.get('mode')):
                raise ValueError(f'Joint checkpoint mixes epochs or modes: {key}')
            bundle_consumed.add(key)
        module, meta = load_variant(str(path), device=device, payload=payload)
        if (meta.get('kind') != kind or meta.get('method') != method
                or not fingerprint or meta.get('run_fingerprint') != fingerprint):
            raise ValueError(f'Checkpoint method/run mismatch: {path}')
        module.requires_grad_(False)
        return module, meta

    for index, concept in enumerate(concepts):
        labels = concept.get('tnls_dl3', []) if is_json else [index + 1]
        for position, text in enumerate(concept['prompts']):
            record = dict(prompt_index=len(records), prompt=text,
                          concept_index=index, concept_id=concept['concept_id'],
                          prompt_position=position, valid_labels=list(labels))
            records.append(record)
            if index not in selected:
                continue
            path = directory / prompt_folder / f'prompt_{record["prompt_index"]:03d}' / 'best.pth'
            module, meta = checked_load(path, 'prompt', cx['prompt_method'])
            identity = meta.get('identity', {})
            if any(identity.get(key) != value for key, value in record.items()):
                raise ValueError(f'Prompt identity mismatch: {path}')
            prompts[record['prompt_index']] = module
        if index not in selected:
            continue
        matches = list((directory / 'concept').glob(f'concept_{index:03d}_*/best.pth'))
        if len(matches) != 1:
            raise ValueError(f'Expected one best.pth for concept {index}; found {matches}')
        module, meta = checked_load(matches[0], 'concept', cx['concept_method'])
        members = [r for r in records if r['concept_index'] == index]
        saved_members = meta.get('members', [])
        if (meta.get('prompt_count') != len(members) or len(saved_members) != len(members)
                or any(any(saved.get(k) != v for k, v in member.items())
                       for saved, member in zip(saved_members, members))):
            raise ValueError(f'Concept membership mismatch: {matches[0]}')
        aggregators[index] = module
    if consumed != set(overrides):
        raise ValueError(f'Unknown checkpoint override slots: {sorted(set(overrides)-consumed)}')
    if bundle is not None and bundle_consumed != set(bundle['modules']):
        raise ValueError('Joint checkpoint contains unknown/extra modules')
    effective = dict(cx)
    if joint_refiner is not None:
        effective['joint_refiner'] = joint_refiner
    if refiner_strength is not None:
        effective['joint_refiner_strength'] = refiner_strength
    method, strength = refiner_settings(effective)
    if method != 'none':
        if bundle is None or bundle['format'] != 'cx_b_joint_v2':
            raise ValueError('Enabling a refiner requires a saved R1 checkpoint')
        aggregators.refiner = load_refiner(bundle['refiner'], device, strength=strength)
        aggregators.refiner.requires_grad_(False)
        aggregators.background = config['cx_b']['joint_train']['background']
    return run, concepts, selected, records, prompts, aggregators


def collect_samples(config, args):
    key = {'val': 'validation_dataloader', 'train': 'train_dataloader',
           'test': 'test_dataloader'}[args.split]
    if key not in config:
        if not (args.image_dir and args.gt_dir):
            raise ValueError(f'No {key} in snapshot; provide --image-dir and --gt-dir.')
        dataset = {}
    else:
        dataset = config[key]['dataset']
    if 'datasets' in dataset or 'dataset' in dataset:
        raise ValueError('Wrapped datasets are unsupported; provide a flat dataset configuration.')
    if dataset.get('reduce_zero_label', False):
        raise ValueError('Expected original GT IDs; reduce_zero_label=True is unsupported.')
    root = resolve_path(args.data_root or dataset.get('data_root', '.'))
    prefix = dataset.get('data_prefix', {})
    image_dir = resolve_path(args.image_dir) if args.image_dir else root / prefix['img_path']
    gt_dir = resolve_path(args.gt_dir) if args.gt_dir else root / prefix['seg_map_path']
    paths = sorted(p for p in image_dir.rglob('*') if p.is_file() and (
        p.name.endswith(dataset['img_suffix']) if dataset.get('img_suffix')
        else p.suffix.lower() in IMAGE_SUFFIXES))
    samples = []
    for image in paths:
        relative = image.relative_to(image_dir)
        candidates = ([relative.with_suffix(dataset['seg_map_suffix'])]
                      if dataset.get('seg_map_suffix') else [])
        candidates += [relative] + [relative.with_suffix(s) for s in IMAGE_SUFFIXES]
        gt = next((gt_dir / p for p in candidates if (gt_dir / p).is_file()), None)
        # The training cache enumerates the sorted set of paired images only.
        if gt is not None:
            samples.append((image, gt, relative))
    if not samples:
        raise ValueError(f'No paired images and GT found: {image_dir}, {gt_dir}')
    if len(samples) != len(paths):
        raise ValueError(f'{len(paths) - len(samples)} images have no GT in {gt_dir}.')
    return samples, dataset


class SavedHeadCache:
    """Read the training cache without importing or constructing SAM/trainers."""
    def __init__(self, root, split, model_config, device):
        self.root, self.split, self.device = resolve_path(root), split, device
        self.signature = dict(
            confidence_threshold=float(model_config.get('confidence_threshold', 0.5)),
            use_sem_seg=bool(model_config.get('use_sem_seg', True)),
            use_transformer_decoder=bool(model_config.get('use_transformer_decoder', True)),
            instance_filter_use_presence=bool(model_config.get('instance_filter_use_presence', True)))

    def load(self, record, sample_index, image):
        prompt_index = record['prompt_index']
        # Match the training filename sanitizer exactly (one '_' per whitespace).
        stem = ''.join('_' if c.isspace() or c in '<>:"/\\|?*' or ord(c) < 32
                       else c for c in image.stem).strip(' ._')[:72].rstrip(' ._') or 'item'
        path = self.root / self.split / f'prompt_{prompt_index:03d}' / f'{sample_index:06d}_{stem}.pth'
        if not path.is_file():
            raise FileNotFoundError(f'Missing frozen head cache: {path}. '
                                    'Use the matching dataset/cache or --prompt-head-cache.')
        cached = torch.load(str(path), map_location='cpu')
        expected = dict(image_path=os.path.abspath(image), prompt_idx=prompt_index,
                        prompt=record['prompt'], cache_signature=self.signature)
        if any(cached.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Head cache identity/settings mismatch: {path}')
        return {k: v.to(device=self.device, dtype=torch.float32)
                for k, v in cached['heads'].items()}


def predict_image(records, selected, prompts, aggregators, get_heads, online=None,
                  prompt_callback=None):
    """Identical prompt -> per-concept aggregation path to CX-B training."""
    concept_scores, prompt_scores = [], {}
    refiner = getattr(aggregators, 'refiner', None)
    if refiner is not None and list(selected) != sorted(aggregators):
        raise ValueError('A shared refiner must see the full query set')
    for index in selected:
        members = [r for r in records if r['concept_index'] == index]
        heads = [get_heads(r) for r in members]
        probabilities = [prompts[r['prompt_index']](h) for r, h in zip(members, heads)]
        for record, score, head in zip(members, probabilities, heads):
            prompt_scores[record['prompt_index']] = score.detach().cpu().numpy()
            if prompt_callback is not None:
                prompt_callback(record, head, score)
        probability = aggregators[index](
            torch.stack(probabilities),
            presence=torch.stack([torch.as_tensor(h['presence_score']).squeeze() for h in heads]),
            heads={k: torch.stack([h[k] for h in heads])
                   for k in ['semantic_score_map', 'instance_score_map']},
            regions=[online[r['prompt_index']]['regions'] if online else None for r in members])
        concept_scores.append(probability if refiner is not None else probability.detach().cpu().numpy())
    if refiner is not None:
        raw = torch.stack(concept_scores)
        aggregators.raw_scores = raw.detach().cpu().numpy()
        return refiner(raw, background=aggregators.background).detach().cpu().numpy(), prompt_scores
    return np.stack(concept_scores), prompt_scores


def classify_scores(scores, selected, threshold, background=NODATA):
    if scores.ndim != 3 or not np.isfinite(scores).all():
        raise ValueError('Expected finite concept probabilities with shape [C,H,W].')
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('Concept probabilities outside [0,1].')
    prediction = np.asarray(selected, dtype=np.uint16)[scores.argmax(axis=0)]
    prediction[scores.max(axis=0) < threshold] = background
    return prediction


def map_gt(gt, concepts, is_json, ignore_index):
    mapped = np.full(gt.shape, NODATA, dtype=np.uint16)
    valid = gt != ignore_index
    for index, concept in enumerate(concepts):
        labels = concept.get('tnls_dl3', []) if is_json else [index + 1]
        mask = np.isin(gt, labels) & valid
        if np.any(mask & (mapped != NODATA)):
            raise ValueError('Overlapping GT label groups cannot form a multiclass label map.')
        mapped[mask] = index
    return mapped


def export_ground_truth(output, key, gt, mapped_gt, concepts, is_json,
                        ignore_index, active_records):
    """Export GT independently of visualization flags; retain raw label dtype."""
    valid = gt != ignore_index
    masks = np.stack([
        np.isin(gt, c.get('tnls_dl3', []) if is_json else [i + 1]) & valid
        for i, c in enumerate(concepts)])
    prompt_masks = np.stack([masks[r['concept_index']] for r in active_records])
    paths = dict(
        raw=output / 'data/gt/raw' / Path(str(key) + '.npy'),
        concept_labels=output / 'data/gt/concept_labels' / Path(str(key) + '.png'),
        valid_mask=output / 'data/gt/valid' / Path(str(key) + '.png'))
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths['raw'], gt, allow_pickle=False)
    Image.fromarray(mapped_gt).save(paths['concept_labels'])
    Image.fromarray(valid.astype(np.uint8) * 255).save(paths['valid_mask'])
    concept_files = []
    for index, (concept, mask) in enumerate(zip(concepts, masks)):
        name = concept.get('concept_name') or concept['concept_id']
        path = output / 'data/gt/concept' / key / f'{index:03d}_{safe_filename(name)}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        # Distinguish an ignored pixel from a genuine negative pixel.
        encoded = np.where(valid, mask.astype(np.uint8), 255).astype(np.uint8)
        Image.fromarray(encoded).save(path)
        concept_files.append(dict(concept_index=index, path=str(path.relative_to(output))))
    arrays = dict(gt_raw=gt, gt_concept=mapped_gt, gt_valid_mask=valid,
                  gt_concept_masks=masks,
                  gt_concept_indices=np.arange(len(concepts), dtype=np.int32),
                  gt_prompt_masks=prompt_masks)
    files = {name: str(path.relative_to(output)) for name, path in paths.items()}
    files['concept_masks'] = concept_files
    return arrays, files


def metric_rows(counts, names):
    rows = []
    for name, (tp, fp, fn) in zip(names, counts):
        divide = lambda a, b: float(a / b) if b else None
        rows.append(dict(Class=name, TP=int(tp), FP=int(fp), FN=int(fn),
                         IoU=divide(tp, tp + fp + fn),
                         Precision=divide(tp, tp + fp), Recall=divide(tp, tp + fn),
                         Fscore=divide(2 * tp, 2 * tp + fp + fn)))
    return rows


def diagnostic_prompt_panel(record, module, heads, score, raster, gt, valid,
                             output_path, threshold):
    """Prepare model values; all figure rendering stays in visual_utils."""
    base = getattr(module, 'base', module)
    semantic = heads['semantic_score_map'].float()
    instance = heads['instance_score_map'].float()
    presence = heads['presence_score'].float().squeeze()
    if isinstance(module, SegOV3PromptAdapter):
        fused = torch.maximum(semantic, instance)
        gated = instance * presence
    elif hasattr(module, 'base'):
        # Recover the spatial fusion of a DynamicPromptHead before its final
        # presence offset. The final probability is always the actual output.
        offset = base.presence_gain * (presence - 0.5) + base.presence_intercept
        fused = torch.sigmoid(torch.logit(score.clamp(1e-7, 1 - 1e-7)) - offset)
        gated = torch.sigmoid(torch.logit(instance.clamp(1e-6, 1 - 1e-6)) + offset)
    else:
        fused = base.semantic_weight * semantic + (1 - base.semantic_weight) * instance
        offset = base.presence_gain * (presence - 0.5) + base.presence_intercept
        gated = torch.sigmoid(torch.logit(instance.clamp(1e-6, 1 - 1e-6)) + offset)
    array = lambda value: value.detach().float().cpu().numpy()
    final = array(score)
    target = np.isin(gt, record['valid_labels']) & valid
    prediction = (final >= threshold) & valid
    tp = np.count_nonzero(prediction & target)
    fp = np.count_nonzero(prediction & ~target)
    fn = np.count_nonzero(~prediction & target)
    tn = np.count_nonzero(valid & ~prediction & ~target)
    row = metric_rows([(tp, fp, fn)], [record['prompt']])[0]
    background_iou = tn / (tn + fp + fn) if tn + fp + fn else None
    ious = [v for v in [row['IoU'], background_iou] if v is not None]
    metrics = dict(iou=row['IoU'], precision=row['Precision'], recall=row['Recall'],
                   fscore=row['Fscore'], miou=float(np.mean(ious)) if ious else None,
                   label_area_ratio=float(target.sum() / valid.sum()) if valid.any() else None,
                   pred_area_ratio=float(prediction.sum() / valid.sum()) if valid.any() else None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_panel(Image.fromarray(raster), array(semantic), array(instance),
               array(gated), array(fused), final, float(presence.cpu()),
               record['prompt'], output_path, gt_mask=target,
               pred_mask=prediction, final_metrics=metrics,
               prediction_threshold=threshold)


def build_head_source(run, records, args, device):
    """Construct the saved run's raw-head source, shared by both evaluators."""
    config, cx = run['config'], run['config']['cx_b']
    if cx['data_mode'] == 'cache':
        cache = SavedHeadCache(args.prompt_head_cache or run.get('inputs', {}).get(
            'prompt_head_cache') or cx['prompt_head_cache'], args.split, config['model'], device)
        return cache, None
    if cx['data_mode'] != 'online_sam':
        raise ValueError(f'Unsupported data_mode: {cx["data_mode"]}')
    if device.type != 'cuda':
        raise ValueError('The existing online SAM model requires CUDA. Cached evaluation supports CPU.')
    os.environ.setdefault('SEGOV3_LIGHTWEIGHT_TRAIN', '1')
    from segearthov3_learnable_segmentor import SegEarthOV3LearnableSegmentation
    from cx_b.heads import OnlineSAMHeadProvider
    kwargs = dict(config['model'])
    kwargs.pop('type', None)
    kwargs.pop('model_type', None)
    kwargs['classname_path'] = str(resolve_path(kwargs['classname_path']))
    kwargs['device'] = device
    model = SegEarthOV3LearnableSegmentation(**kwargs).eval()
    model.requires_grad_(False)
    if list(model.query_words) != [r['prompt'] for r in records]:
        raise ValueError('Current concept file differs from the saved prompt order.')
    return None, OnlineSAMHeadProvider(model, cx['sam_region_cfg'])


def run_evaluation(args):
    device = torch.device(args.device)
    run, concepts, selected, records, prompts, aggregators = load_run(args.pth_dir, device, **evaluation_refiner_options(args))
    config, cx = run['config'], run['config']['cx_b']
    samples, dataset = collect_samples(config, args)
    total = len(samples) if args.max_images < 0 else min(args.max_images, len(samples))
    binary_threshold = args.threshold if args.threshold is not None else cx.get('metric_threshold', 0.5)
    pred_threshold = (args.prediction_threshold if args.prediction_threshold is not None
                      else config['model'].get('prob_thd', 0.1))
    prediction_background = cx.get('prediction_background', NODATA)
    if prediction_background != NODATA and prediction_background not in range(len(concepts)):
        raise ValueError('prediction_background must be a defined concept index or 65535.')
    is_json = str(config.get('concept_path', config['model']['classname_path'])).lower().endswith('.json')
    ignore_index = dataset.get('ignore_index', 255) if is_json else 0
    names = [c.get('concept_name') or c['concept_id'] for c in concepts]
    palette = config.get('concept_palette') or [concept_color(i).tolist() for i in range(len(concepts))]
    if len(palette) != len(concepts):
        raise ValueError('Palette length does not match concept definitions.')
    output = Path(args.out).expanduser().resolve()
    if ablation_folder(args):
        output = output/ablation_folder(args)
    source = Path(args.pth_dir).expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Use an output directory separate from the checkpoint tree.')
    for folder in ['data', 'panel', 'vis/out']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    active_records = [r for r in records if r['concept_index'] in selected]
    cache, provider = build_head_source(run, records, args, device)
    counts = np.zeros((len(selected), 3), dtype=np.int64)
    multiclass_counts = np.zeros_like(counts)
    refined_counts = np.zeros_like(counts) if aggregators.refiner is not None else None
    changes = dict(changed_pixels=0, corrected_pixels=0, damaged_pixels=0)
    manifest = []
    print(f'CX-B {cx["prompt_method"]}/{cx["concept_method"]}: '
          f'{len(selected)} concepts, {len(active_records)} prompts, {total} images', flush=True)
    with torch.inference_mode():
        for sample_index, (image_path, gt_path, relative) in enumerate(samples[:total]):
            with Image.open(gt_path) as image:
                gt = np.asarray(image).copy()
            if gt.ndim != 2:
                raise ValueError(f'GT must be a single-channel label raster: {gt_path}')
            with Image.open(image_path) as image:
                raster = np.asarray(image.convert('RGB')).copy()
            if raster.shape[:2] != gt.shape:
                raise ValueError(f'Image and GT shapes differ: {image_path}')
            key = relative.parent / relative.name
            valid = gt != ignore_index

            def panel_callback(record, heads, score):
                if score.shape != gt.shape:
                    raise ValueError(f'Head and GT shapes differ: {image_path}')
                prompt_index = record['prompt_index']
                diagnostic_prompt_panel(
                    record, prompts[prompt_index], heads, score, raster, gt, valid,
                    output / 'panel' / key / f'{prompt_index:03d}_{safe_filename(record["prompt"])}.png',
                    binary_threshold)

            online = provider.image_outputs(image_path, [r['prompt_index'] for r in active_records]) if provider else None
            get_heads = (lambda r: online[r['prompt_index']]['heads']) if online else (
                lambda r: cache.load(r, sample_index, image_path))
            scores, prompt_scores = predict_image(
                records, selected, prompts, aggregators, get_heads, online,
                prompt_callback=panel_callback)
            if scores.shape[1:] != gt.shape:
                raise ValueError(f'Head and GT shapes differ: {image_path}')
            prediction = classify_scores(
                scores, selected, pred_threshold, prediction_background)
            mapped_gt = map_gt(gt, concepts, is_json, ignore_index)
            if aggregators.refiner is not None:
                raw_prediction = classify_scores(aggregators.raw_scores, selected, pred_threshold, prediction_background)
                for key_name, value in prediction_changes(raw_prediction, prediction, mapped_gt, valid & (mapped_gt != NODATA)).items():
                    changes[key_name] += value
            gt_arrays, gt_files = export_ground_truth(
                output, key, gt, mapped_gt, concepts, is_json, ignore_index, active_records)
            # Keep original relative folders and extension to prevent name collisions.
            data_path = output / 'data' / key
            data_path.parent.mkdir(parents=True, exist_ok=True)
            label_path = Path(str(data_path) + '.png')
            score_path = Path(str(data_path) + '.npz')
            Image.fromarray(prediction).save(label_path)
            np.savez_compressed(
                score_path, prediction=prediction, concept_scores=scores.astype(np.float32),
                concept_indices=np.array(selected, dtype=np.int32),
                prompt_scores=np.stack([prompt_scores[r['prompt_index']] for r in active_records]).astype(np.float32),
                prompt_indices=np.array([r['prompt_index'] for r in active_records], dtype=np.int32),
                **gt_arrays, **({'raw_concept_scores': aggregators.raw_scores.astype(np.float32)}
                               if aggregators.refiner is not None else {}))
            gt_visual_path = output / 'vis/gt' / Path(str(key) + '.png')
            gt_visual_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(render_labeled_overlay(
                raster, mapped_gt, names, palette, args.alpha, NODATA)).save(gt_visual_path)
            save_labeled_comparison(
                raster, mapped_gt, prediction, names, palette,
                output / 'vis/out' / Path(str(key) + '.png'), args.alpha, NODATA)
            for channel, index in enumerate(selected):
                labels = concepts[index].get('tnls_dl3', []) if is_json else [index + 1]
                target = np.isin(gt, labels) & valid
                raw_scores = aggregators.raw_scores if aggregators.refiner is not None else scores
                binary = (raw_scores[channel] >= binary_threshold) & valid
                destinations = [(counts, binary), (multiclass_counts, (prediction == index) & valid)]
                if refined_counts is not None:
                    destinations.append((refined_counts, (scores[channel] >= binary_threshold) & valid))
                for destination, mask in destinations:
                    destination[channel] += (np.count_nonzero(mask & target),
                                              np.count_nonzero(mask & ~target),
                                              np.count_nonzero(~mask & target))
                if args.vis_concept:
                    save_binary_comparison(
                        raster, target, scores[channel], names[index], palette[index],
                        output / 'vis/concept' / key / f'{index:03d}_{safe_filename(names[index])}.png',
                        binary_threshold, args.alpha, valid)
            if args.vis_prompt:
                for record in active_records:
                    index, prompt_index = record['concept_index'], record['prompt_index']
                    save_binary_comparison(
                        raster, np.isin(gt, record['valid_labels']) & valid,
                        prompt_scores[prompt_index], record['prompt'], palette[index],
                        output / 'vis/prompt' / key / f'{prompt_index:03d}_{safe_filename(record["prompt"])}.png',
                        binary_threshold, args.alpha, valid)
            manifest.append(dict(image=str(image_path), gt=str(gt_path),
                                 gt_outputs=gt_files,
                                 gt_visualization=str(gt_visual_path.relative_to(output)),
                                 prediction=str(label_path.relative_to(output)),
                                 scores=str(score_path.relative_to(output))))
            print(f'[{sample_index + 1}/{total}] {relative}', flush=True)
    rows = metric_rows(counts, [names[i] for i in selected])
    multiclass = metric_rows(multiclass_counts, [names[i] for i in selected])
    mean = lambda items: float(np.mean([r['IoU'] for r in items if r['IoU'] is not None])) if any(r['IoU'] is not None for r in items) else None
    summary = dict(scale='fraction', images=total,
                   independent_binary=dict(mIoU=mean(rows), per_concept=rows),
                   multiclass_argmax=dict(mIoU=mean(multiclass), per_concept=multiclass))
    if refined_counts is not None:
        refined_rows = metric_rows(refined_counts, [names[i] for i in selected])
        summary.update(refined_binary=dict(mIoU=mean(refined_rows), per_concept=refined_rows),
                       refiner_changes=changes)
    metadata = dict(checkpoint_dir=str(source), fingerprint=run['fingerprint'],
                    split=args.split, binary_threshold=binary_threshold,
                    prediction_threshold=pred_threshold, prediction_nodata=NODATA,
                    prediction_background=prediction_background,
                    prediction_encoding='zero-based original concept index; 65535=unassigned',
                    gt_ignore_index=ignore_index, concepts=concepts,
                    gt_encoding=dict(
                        raw='original dataset labels and dtype',
                        concept_labels='zero-based concept index; 65535=ignored or unmapped',
                        concept_mask_png='0=negative; 1=positive; 255=ignored',
                        valid_mask_png='0=ignored; 255=valid',
                        npz_masks='boolean; ignored pixels are False; use gt_valid_mask',
                        concept_channels='gt_concept_indices (all defined concepts)',
                        prompt_channels='prompt_indices (GT of the owning concept)'),
                    selected_concepts=selected, prompts=active_records, palette=palette,
                    vis_concept=args.vis_concept, vis_prompt=args.vis_prompt, files=manifest)
    if refiner_settings(cx)[0] != 'none' or ablation_folder(args):
        metadata['refiner'] = dict(method='none' if aggregators.refiner is None else aggregators.refiner.method,
                                  strength=0 if aggregators.refiner is None else aggregators.refiner.strength,
                                  overrides=evaluation_refiner_options(args),
                                  independent_binary='raw',
                                  concept_scores='refined' if aggregators.refiner is not None else 'raw')
    for filename, data in [('results.json', summary), ('manifest.json', metadata)]:
        (output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Finished: {output}', flush=True)
    return summary


if __name__ == '__main__':
    run_evaluation(parse_args())
