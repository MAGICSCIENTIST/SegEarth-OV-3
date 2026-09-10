import argparse
import csv
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path

os.environ.setdefault('SEGOV3_LIGHTWEIGHT_TRAIN', '1')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from concept_utils import is_json_path, load_concepts
from metrics_reporting import MetricsReporter
import train_learnable_params as shared
from diagnose_sam3_heads import (
    binary_metrics,
    save_gt_overview,
    save_normal_panel,
    save_panel,
)


DEFAULTS = {
    'epochs_per_prompt': 80,
    'lr': 3e-3,
    'patience': 12,
    'validate_every_epochs': 1,
    'max_train_samples': -1,
    'max_val_samples': -1,
    'metric_threshold': 0.5,
    'bce_loss_weight': 0.0,
    'soft_iou_loss_weight': 1.0,
    'recall_loss_weight': 0.1,
    'recall_target': 0.3,
    'max_positive_weight': 20.0,
    'param_reg_weight': 0.01,
    'grad_clip_norm': 1.0,
    'presence_gain_max': 8.0,
    'prompt_weight_max': 3.0,
    'cache_dtype': 'float16',
    'progress_interval': 20,
    'seed': 3407,
    'selection_metric': 'soft_iou',
    'val_visualize_every_epochs': 1,
    'visual_type': 'full_panel',
    'experiment_id': 'A3-0',
    'identity_fallback': False,
    'identity_fallback_min_delta': 0.005,
    'selection_tie_breaker': 'earliest_epoch',
    'reject_baseline_iou_threshold': 0.0,
    'prompt_parameterization': 'four_parameter',
    'prompt_stratified_sampling': False,
    'prompt_group_ratio': (4.0, 2.0, 3.0, 1.0),
    'min_gt_positive_pixels': 1,
    'min_response_positive_pixels': 1,
    'empty_group_policy': 'renormalize',
    'samples_per_epoch': -1,
    'loss_name': 'soft_iou',
    'tversky_alpha_fp': 0.6,
    'tversky_beta_fn': 0.4,
    'cluster_min_hard_iou_gain': 0.005,
    'cluster_min_size': 3,
    'cluster_max_k': 8,
    'cluster_min_ari': 0.8,
    'prompt_hard_iou_std_max': 0.01,
}

REQUIRED_RESULT_FIELDS = {
    'baseline_epoch',
    'baseline_metrics',
    'raw_best_epoch',
    'raw_best_metrics',
    'selected_epoch',
    'selected_metrics',
    'fallback_applied',
    'fallback_reason',
    'prompt_exit',
    'seed',
    'config_snapshot',
}


class TeeStream:
    """Mirror terminal output to a persistent experiment log."""

    def __init__(self, stream, *log_files):
        self.stream = stream
        self.log_files = log_files

    def write(self, text):
        self.stream.write(text)
        for log_file in self.log_files:
            log_file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        for log_file in self.log_files:
            log_file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def prompt_filename(prompt, max_length=80):
    """Keep the real prompt text while replacing Windows-invalid characters."""
    text = ''.join(
        '_' if char in '<>:"/\\|?*' or ord(char) < 32 else char
        for char in str(prompt))
    text = text.strip().rstrip('.') or 'prompt'
    return text[:max_length]


def parse_args():
    """Parse CLI options for cache construction and prompt-wise training."""
    parser = argparse.ArgumentParser(
        description=(
            'Train one independent calibration parameter group per SAM3 '
            'prompt. Only the active prompt is passed to the optimizer.'))
    parser.add_argument('config')
    parser.add_argument('--save-dir', default='weights/prompt_independent')
    parser.add_argument(
        '--prompt-head-cache',
        required=True,
        help=(
            'Directory containing cached SAM3 prompt heads. It is built only '
            'when the directory does not exist.'))
    parser.add_argument('--epochs-per-prompt', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--validate-every-epochs', type=int, default=None)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    parser.add_argument('--metric-threshold', type=float, default=None)
    parser.add_argument('--bce-loss-weight', type=float, default=None)
    parser.add_argument('--soft-iou-loss-weight', type=float, default=None)
    parser.add_argument('--recall-loss-weight', type=float, default=None)
    parser.add_argument('--recall-target', type=float, default=None)
    parser.add_argument('--max-positive-weight', type=float, default=None)
    parser.add_argument('--param-reg-weight', type=float, default=None)
    parser.add_argument('--grad-clip-norm', type=float, default=None)
    parser.add_argument('--presence-gain-max', type=float, default=None)
    parser.add_argument('--prompt-weight-max', type=float, default=None)
    parser.add_argument(
        '--cache-dtype', choices=['float16', 'float32'], default=None)
    parser.add_argument('--progress-interval', type=int, default=None)
    parser.add_argument(
        '--val-visualize-every-epochs',
        type=int,
        default=None,
        help=(
            'Save validation panels every N validation epochs. Validation '
            'metrics are still computed every --validate-every-epochs.'))
    parser.add_argument(
        '--visualize-every-epochs',
        dest='val_visualize_every_epochs',
        type=int,
        help=argparse.SUPPRESS)
    parser.add_argument(
        '--visual-typ',
        '--visual_type',
        dest='visual_type',
        choices=['full_panel', 'normal'],
        default=None,
        help=(
            'Validation image layout: full_panel keeps the diagnostic grid; '
            'normal saves only GT and final-prediction raster overlays.'))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--experiment-id', default=None)
    parser.add_argument(
        '--selection-metric',
        choices=['soft_iou', 'iou', 'hard_iou', 'loss'],
        default=None,
        help='Validation metric used for best checkpoint and early stopping.')
    parser.add_argument(
        '--identity-fallback',
        action=argparse.BooleanOptionalAction,
        default=None)
    parser.add_argument('--identity-fallback-min-delta', type=float, default=None)
    parser.add_argument(
        '--selection-tie-breaker',
        choices=['earliest_epoch'],
        default=None)
    parser.add_argument(
        '--reject-baseline-iou-threshold', type=float, default=None)
    parser.add_argument(
        '--prompt-parameterization',
        choices=['four_parameter', 'identifiable_three'],
        default=None)
    parser.add_argument(
        '--prompt-stratified-sampling',
        action=argparse.BooleanOptionalAction,
        default=None)
    parser.add_argument(
        '--prompt-group-ratio', nargs=4, type=float, default=None,
        metavar=('A', 'B', 'C', 'D'))
    parser.add_argument('--min-gt-positive-pixels', type=int, default=None)
    parser.add_argument('--min-response-positive-pixels', type=int, default=None)
    parser.add_argument(
        '--empty-group-policy', choices=['renormalize'], default=None)
    parser.add_argument('--samples-per-epoch', type=int, default=None)
    parser.add_argument(
        '--loss-name', choices=['soft_iou', 'bce_tversky'], default=None)
    parser.add_argument('--tversky-alpha-fp', type=float, default=None)
    parser.add_argument('--tversky-beta-fn', type=float, default=None)
    parser.add_argument('--cluster-min-hard-iou-gain', type=float, default=None)
    parser.add_argument('--cluster-min-size', type=int, default=None)
    parser.add_argument('--cluster-max-k', type=int, default=None)
    parser.add_argument('--cluster-min-ari', type=float, default=None)
    parser.add_argument('--prompt-hard-iou-std-max', type=float, default=None)
    parser.add_argument('--prompt-indices', nargs='*', type=int, default=None)
    parser.add_argument(
        '--rebuild-cache',
        action='store_true',
        help='Explicitly overwrite cache entries even when the directory exists.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--cfg-options', nargs='+', action='append', default=None)
    return parser.parse_args()


def apply_config(args, cfg):
    """Fill unset CLI values from prompt_independent_train_cfg defaults."""
    train_cfg = cfg.get('prompt_independent_train_cfg', {})
    for key, default in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, train_cfg.get(key, default))
    if args.epochs_per_prompt < 0:
        raise ValueError('epochs_per_prompt must be non-negative.')
    if args.validate_every_epochs <= 0:
        raise ValueError('validate_every_epochs must be positive.')
    if not 0.0 <= args.metric_threshold <= 1.0:
        raise ValueError('metric_threshold must be in [0, 1].')
    if args.presence_gain_max <= 0 or args.prompt_weight_max <= 0:
        raise ValueError('Prompt parameter maxima must be positive.')
    if args.val_visualize_every_epochs < 0:
        raise ValueError(
            'val_visualize_every_epochs must be non-negative.')
    if args.visual_type not in ('full_panel', 'normal'):
        raise ValueError(
            "visual_type must be either 'full_panel' or 'normal'.")
    if any(value < 0 for value in args.prompt_group_ratio):
        raise ValueError('prompt_group_ratio values must be non-negative.')
    if sum(args.prompt_group_ratio) <= 0:
        raise ValueError('prompt_group_ratio must contain a positive value.')
    if args.min_gt_positive_pixels < 0:
        raise ValueError('min_gt_positive_pixels must be non-negative.')
    if args.min_response_positive_pixels < 0:
        raise ValueError('min_response_positive_pixels must be non-negative.')
    if args.samples_per_epoch == 0 or args.samples_per_epoch < -1:
        raise ValueError('samples_per_epoch must be -1 or positive.')
    if args.tversky_alpha_fp < 0 or args.tversky_beta_fn < 0:
        raise ValueError('Tversky alpha/beta must be non-negative.')
    if args.tversky_alpha_fp + args.tversky_beta_fn <= 0:
        raise ValueError('At least one Tversky weight must be positive.')
    if args.cluster_min_size < 2 or args.cluster_max_k < 2:
        raise ValueError('Cluster sizes and limits must be at least 2.')
    if not -1.0 <= args.cluster_min_ari <= 1.0:
        raise ValueError('cluster_min_ari must be in [-1, 1].')
    if args.prompt_hard_iou_std_max < 0:
        raise ValueError('prompt_hard_iou_std_max must be non-negative.')
    cfg.model['prompt_parameterization'] = args.prompt_parameterization


def limited_length(dataset, max_samples):
    """Return the effective sample count after applying an optional limit."""
    if max_samples is None or max_samples < 0:
        return len(dataset)
    return min(len(dataset), max_samples)


def cache_file(cache_dir, split, prompt_idx, sample_idx, image_path):
    """Build a stable cache path for one image and one prompt."""
    stem = shared.safe_filename(osp.splitext(osp.basename(image_path))[0], 72)
    return osp.join(
        cache_dir,
        split,
        f'prompt_{prompt_idx:03d}',
        f'{sample_idx:06d}_{stem}.pth')


def cache_heads(heads, dtype):
    """Detach frozen SAM3 heads and move them to compact CPU storage."""
    cache_dtype = torch.float16 if dtype == 'float16' else torch.float32
    return {
        name: value.detach().to(device='cpu', dtype=cache_dtype)
        for name, value in heads.items()
    }


def cache_signature(model):
    """Describe model settings that affect frozen head extraction."""
    return dict(
        confidence_threshold=float(model.confidence_threshold),
        use_sem_seg=bool(model.use_sem_seg),
        use_transformer_decoder=bool(model.use_transformer_decoder),
        instance_filter_use_presence=bool(
            model.instance_filter_use_presence))


def build_head_cache(model, dataset, split, prompt_indices, args):
    """Cache frozen heads while sharing image encoding across missing prompts.

    The cache contains no learnable calibration output. This lets later epochs
    optimize prompt scalars without executing the SAM3 backbone again.
    """
    target_samples = limited_length(
        dataset,
        args.max_train_samples if split == 'train' else args.max_val_samples)
    started = time.time()
    generated = 0
    for sample_idx in range(target_samples):
        sample = dataset[sample_idx]
        data_sample = sample['data_samples']
        image_path = data_sample.metainfo['img_path']
        missing = []
        for prompt_idx in prompt_indices:
            path = cache_file(
                args.prompt_head_cache,
                split,
                prompt_idx,
                sample_idx,
                image_path)
            if args.rebuild_cache or not osp.exists(path):
                missing.append(prompt_idx)
        if not missing:
            continue

        gt = data_sample.gt_sem_seg.data.squeeze()
        image = Image.open(image_path).convert('RGB')
        all_heads = model.extract_prompt_heads_batch(
            image, missing, ori_shape=tuple(gt.shape[-2:]))
        for prompt_idx, heads in zip(missing, all_heads):
            path = cache_file(
                args.prompt_head_cache,
                split,
                prompt_idx,
                sample_idx,
                image_path)
            os.makedirs(osp.dirname(path), exist_ok=True)
            torch.save(dict(
                image_path=osp.abspath(image_path),
                prompt_idx=prompt_idx,
                prompt=model.query_words[prompt_idx],
                cache_signature=cache_signature(model),
                heads=cache_heads(heads, args.cache_dtype)), path)
            generated += 1

        done = sample_idx + 1
        if (args.progress_interval > 0
                and (done % args.progress_interval == 0
                     or done == target_samples)):
            elapsed = time.time() - started
            print(
                f'[cache:{split}] {done}/{target_samples} images '
                f'generated={generated} elapsed={shared.format_duration(elapsed)}',
                flush=True)

    manifest = dict(
        split=split,
        samples=target_samples,
        prompt_indices=prompt_indices,
        cache_dtype=args.cache_dtype,
        query_words=[model.query_words[index] for index in prompt_indices],
        use_sem_seg=model.use_sem_seg,
        use_transformer_decoder=model.use_transformer_decoder,
        instance_filter_use_presence=model.instance_filter_use_presence,
        confidence_threshold=model.confidence_threshold)
    os.makedirs(args.prompt_head_cache, exist_ok=True)
    with open(osp.join(
            args.prompt_head_cache, f'{split}_manifest.json'), 'w',
              encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def prepare_head_cache(model, train_dataset, validation_dataset,
                       prompt_indices, args):
    """Build the cache once, or reuse the supplied directory unchanged."""
    if osp.exists(args.prompt_head_cache) and not osp.isdir(
            args.prompt_head_cache):
        raise ValueError(
            f'prompt_head_cache is not a directory: '
            f'{args.prompt_head_cache}')
    should_build = (
        args.rebuild_cache or not osp.isdir(args.prompt_head_cache))
    if not should_build:
        print(
            f'Reusing existing prompt head cache: '
            f'{args.prompt_head_cache}')
        return False

    build_head_cache(
        model, train_dataset, 'train', prompt_indices, args)
    if validation_dataset is not None:
        build_head_cache(
            model, validation_dataset, 'val', prompt_indices, args)
    return True


def load_heads_cpu(path, expected_image_path, expected_prompt_idx,
                   expected_prompt, expected_signature):
    """Load and validate cached heads without changing their storage format."""
    cached = torch.load(path, map_location='cpu')
    if osp.abspath(expected_image_path) != cached.get('image_path'):
        raise RuntimeError(
            f'Prompt cache image mismatch in {path}. Rebuild the cache.')
    if int(cached.get('prompt_idx', -1)) != int(expected_prompt_idx):
        raise RuntimeError(
            f'Prompt cache index mismatch in {path}. Rebuild the cache.')
    if cached.get('prompt') != expected_prompt:
        raise RuntimeError(
            f'Prompt cache text mismatch in {path}. Rebuild the cache.')
    if cached.get('cache_signature') != expected_signature:
        raise RuntimeError(
            f'Prompt cache model settings mismatch in {path}. '
            'Rebuild the cache.')
    return cached['heads']


def load_heads(path, expected_image_path, expected_prompt_idx,
               expected_prompt, expected_signature, device):
    """Load cached heads and move them to the requested compute device."""
    cached_heads = load_heads_cpu(
        path,
        expected_image_path,
        expected_prompt_idx,
        expected_prompt,
        expected_signature)
    return {
        name: value.to(device=device, dtype=torch.float32)
        for name, value in cached_heads.items()
    }


def binary_target(data_sample, valid_labels, ignore_index, device):
    """Build one prompt's binary GT from its owning concept label union."""
    gt = data_sample.gt_sem_seg.data.squeeze().long().to(device)
    valid_mask = gt != ignore_index
    target = torch.zeros_like(gt, dtype=torch.float32)
    for label in valid_labels:
        target = torch.where(
            gt == int(label), torch.ones_like(target), target)
    return target, valid_mask


def concept_valid_labels(concept, concept_idx, concept_path):
    """Resolve dataset labels for one concept definition source."""
    if is_json_path(concept_path):
        return [int(value) for value in concept.get('tnls_dl3', [])]
    return [int(concept_idx) + 1]


def concept_ignore_index(concept_path, configured_ignore_index):
    """Use label zero as nodata for line-based TXT concept files."""
    if is_json_path(concept_path):
        return int(configured_ignore_index)
    return 0


def prompt_loss(model, prompt_idx, heads, target, valid_mask, args,
                logits=None, tensor_components=False):
    """Compute the active prompt's supervised and parameter regularization loss.

    Optional BCE measures calibration, soft IoU is the stage-one optimization
    target, and the recall floor discourages weak responses on present concepts.
    """
    if logits is None:
        logits = model.calibrate_prompt_head_logits(prompt_idx, heads)
    logits = logits[valid_mask]
    target = target[valid_mask]
    if logits.numel() == 0:
        return None

    positives = target.sum()
    negatives = target.numel() - positives
    has_positives = positives > 0
    positive_weight = torch.where(
        has_positives,
        torch.clamp(
            negatives / positives.clamp_min(1.0),
            min=1.0,
            max=args.max_positive_weight),
        torch.ones((), device=target.device))
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weight)

    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum()
    union = probabilities.sum() + target.sum() - intersection
    positive_soft_iou_loss = 1.0 - (
        intersection + 1e-6) / (union + 1e-6)
    soft_iou_loss = torch.where(
        has_positives, positive_soft_iou_loss, probabilities.mean())
    soft_recall = intersection / positives.clamp_min(1.0)
    recall_loss = torch.where(
        has_positives,
        torch.relu(args.recall_target - soft_recall),
        torch.zeros((), device=target.device))

    true_positive = intersection
    false_positive = (probabilities * (1.0 - target)).sum()
    false_negative = ((1.0 - probabilities) * target).sum()
    tversky_score = (
        (true_positive + 1e-6)
        / (true_positive
           + args.tversky_alpha_fp * false_positive
           + args.tversky_beta_fn * false_negative
           + 1e-6))
    tversky_loss = 1.0 - tversky_score
    metric_loss = (
        tversky_loss if args.loss_name == 'bce_tversky'
        else soft_iou_loss)

    initial_values = model.initial_prompt_parameter_group(prompt_idx)
    regularization = sum(
        (parameter - initial_values[name]).pow(2)
        for name, parameter in model.prompt_parameter_group_items(prompt_idx))
    total = (
        args.bce_loss_weight * bce_loss
        + args.soft_iou_loss_weight * metric_loss
        + args.recall_loss_weight * recall_loss
        + args.param_reg_weight * regularization)
    component_tensors = dict(
        bce=bce_loss.detach(),
        soft_iou=soft_iou_loss.detach(),
        tversky=tversky_loss.detach(),
        recall=recall_loss.detach(),
        regularization=regularization.detach())
    components = (
        component_tensors if tensor_components else {
            name: float(value.cpu())
            for name, value in component_tensors.items()
        })
    return total, probabilities.detach(), target.detach(), components


def empty_metric_state(model):
    """Create pixel-level accumulators for one prompt evaluation pass."""
    return dict(
        loss=0.0,
        samples=0,
        tp=0,
        fp=0,
        fn=0,
        tn=0,
        positive_pixels=0,
        soft_intersection=0.0,
        soft_union=0.0,
        image_iou_sum=0.0,
        image_iou_count=0,
        component_sums=dict(
            bce=0.0,
            soft_iou=0.0,
            tversky=0.0,
            recall=0.0,
            regularization=0.0),
        gradient_sums={
            name: 0.0 for name in model_parameter_names(model)
        },
        gradient_steps=0)


def model_parameter_names(model):
    """Return the scalar prompt parameters tracked in metrics and TensorBoard."""
    return tuple(model.TRAINABLE_PROMPT_PARAMETER_NAMES)


def update_metric_state(state, loss, probabilities, target, threshold,
                        components, gradients=None):
    """Accumulate loss and binary confusion counts for one sample."""
    prediction = probabilities >= threshold
    truth = target > 0.5
    sample_tp = int((prediction & truth).sum().cpu())
    sample_fp = int((prediction & ~truth).sum().cpu())
    sample_fn = int((~prediction & truth).sum().cpu())
    sample_tn = int((~prediction & ~truth).sum().cpu())
    state['loss'] += float(loss)
    state['samples'] += 1
    state['tp'] += sample_tp
    state['fp'] += sample_fp
    state['fn'] += sample_fn
    state['tn'] += sample_tn
    sample_union = sample_tp + sample_fp + sample_fn
    if sample_union > 0:
        state['image_iou_sum'] += sample_tp / sample_union
        state['image_iou_count'] += 1
    state['positive_pixels'] += int(truth.sum().cpu())
    state['soft_intersection'] += float(
        (probabilities * target).sum().cpu())
    state['soft_union'] += float((
        probabilities.sum()
        + target.sum()
        - (probabilities * target).sum()).cpu())
    for name, value in components.items():
        state['component_sums'][name] += float(value)
    if gradients is not None:
        for name, value in gradients.items():
            state['gradient_sums'][name] += float(value)
        state['gradient_steps'] += 1


def finalize_metrics(state):
    """Convert accumulated confusion counts into IoU and related metrics."""
    tp, fp, fn = state['tp'], state['fp'], state['fn']

    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator > 0 else None

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    iou = divide(tp, tp + fp + fn)
    soft_iou = (
        divide(
            state['soft_intersection'] + 1e-6,
            state['soft_union'] + 1e-6)
        if state['samples'] > 0 else None)
    f1 = (
        divide(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None)
    mean_image_iou = divide(
        state['image_iou_sum'], state['image_iou_count'])
    predicted_positive_pixels = tp + fp
    gt_positive_pixels = state['positive_pixels']
    return dict(
        loss=state['loss'] / max(state['samples'], 1),
        loss_total=state['loss'] / max(state['samples'], 1),
        samples=state['samples'],
        positive_pixels=state['positive_pixels'],
        predicted_positive_pixels=predicted_positive_pixels,
        gt_positive_pixels=gt_positive_pixels,
        pred_gt_area_ratio=(
            predicted_positive_pixels / max(gt_positive_pixels, 1)),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=state['tn'],
        iou=iou,
        hard_iou=iou,
        miou=iou,
        mean_image_iou=mean_image_iou,
        soft_iou=soft_iou,
        macc=recall,
        mrecall=recall,
        precision=precision,
        recall=recall,
        f1=f1,
        loss_components={
            name: value / max(state['samples'], 1)
            for name, value in state['component_sums'].items()
        },
        gradients={
            name: value / max(state['gradient_steps'], 1)
            for name, value in state['gradient_sums'].items()
        })


def format_metric(value):
    """Format nullable metrics consistently for terminal and file logs."""
    return 'n/a' if value is None else f'{value:.6f}'


def metrics_text(metrics):
    """Build the complete metric summary printed after each validation."""
    if metrics is None:
        return 'not_run'
    names = (
        ('loss', 'loss'),
        ('hard_iou', 'hard_iou'),
        ('mean_image_iou', 'mean_image_iou'),
        ('soft_iou', 'soft_iou'),
        ('precision', 'precision'),
        ('recall', 'recall'),
        ('f1', 'f1'),
    )
    return ' '.join(
        f'{label}={format_metric(metrics.get(name))}'
        for label, name in names)


def panel_map(value):
    """Convert a cached score tensor into a diagnose-style display map."""
    return value.detach().float().squeeze().clamp(0.0, 1.0).cpu().numpy()


def save_validation_visualization(model, data_sample, heads, prompt_idx,
                                  valid_labels, epoch, args):
    """Save a full diagnostic panel or a compact two-overlay validation view."""
    vis_dir = Path(args.save_dir) / 'vis_data'
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_path = data_sample.metainfo['img_path']
    image_id = shared.safe_filename(osp.splitext(osp.basename(image_path))[0])
    prompt_name = prompt_filename(model.query_words[prompt_idx])
    gt_label = data_sample.gt_sem_seg.data.squeeze().detach().cpu().numpy()
    height, width = gt_label.shape[-2:]
    image = Image.open(image_path).convert('RGB')
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)

    visual_type = getattr(args, 'visual_type', 'full_panel')
    gt_path = vis_dir / f'{image_id}_gt.png'
    if visual_type == 'full_panel' and not gt_path.exists():
        save_gt_overview(
            image=image,
            gt_label=gt_label,
            ignore_index=args.ignore_index,
            output_path=gt_path)

    final_probability = model.calibrate_prompt_heads(prompt_idx, heads)
    final_np = final_probability.detach().float().cpu().numpy()
    valid_mask = gt_label != args.ignore_index
    gt_mask = np.isin(gt_label, valid_labels) & valid_mask
    pred_mask = (final_np >= args.metric_threshold) & valid_mask
    panel_path = vis_dir / (
        f'{image_id}_[{prompt_name}]_[epoch_{epoch:04d}].png')

    prompt = f'{model.query_words[prompt_idx]} | epoch {epoch}'
    if visual_type == 'normal':
        save_normal_panel(
            image=image,
            prompt=prompt,
            output_path=panel_path,
            gt_mask=gt_mask,
            pred_mask=pred_mask)
        return

    params = model._bounded_prompt_values()
    fused = model._fuse_heads(
        prompt_idx,
        heads['instance_score_map'],
        heads['semantic_score_map'],
        params)
    gain = params['presence_gain'][prompt_idx]
    bias = params['presence_bias'][prompt_idx]
    presence_score = heads['presence_score'].detach().float().squeeze()
    presence_logit = (
        gain * (presence_score - bias)).clamp(-8.0, 8.0)
    instance_gated = torch.sigmoid(
        model.probability_to_logits(heads['instance_score_map'])
        + presence_logit)
    final_metrics = binary_metrics(
        score_map=final_np,
        gt_label=gt_label,
        positive_labels=valid_labels,
        threshold=args.metric_threshold,
        ignore_index=args.ignore_index)
    save_panel(
        image=image,
        semantic=panel_map(heads['semantic_score_map']),
        instance_raw=panel_map(heads['instance_score_map']),
        instance_gated=panel_map(instance_gated),
        fused_before_presence=panel_map(fused),
        final_score=final_np,
        presence_score=float(presence_score.cpu()),
        prompt=prompt,
        output_path=panel_path,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        final_metrics=final_metrics,
        prediction_threshold=args.metric_threshold)


def allocate_stratified_counts(groups, ratios, total_samples):
    """Allocate an exact epoch budget across non-empty A/B/C/D groups."""
    group_names = ('A', 'B', 'C', 'D')
    active = [
        name for name, ratio in zip(group_names, ratios)
        if len(groups.get(name, ())) > 0 and ratio > 0
    ]
    if not active:
        raise ValueError('Prompt stratification has no non-empty weighted group.')
    active_weights = np.array([
        ratios[group_names.index(name)] for name in active
    ], dtype=np.float64)
    quotas = active_weights / active_weights.sum() * int(total_samples)
    counts = np.floor(quotas).astype(np.int64)
    remaining = int(total_samples) - int(counts.sum())
    order = np.argsort(-(quotas - counts), kind='stable')
    for index in order[:remaining]:
        counts[index] += 1
    result = {name: 0 for name in group_names}
    result.update({name: int(value) for name, value in zip(active, counts)})
    return result


def stratified_epoch_indices(groups, ratios, total_samples, seed):
    """Draw one deterministic, with-replacement stratified training epoch."""
    rng = np.random.default_rng(seed)
    counts = allocate_stratified_counts(groups, ratios, total_samples)
    chunks = []
    for name in ('A', 'B', 'C', 'D'):
        count = counts[name]
        if count <= 0:
            continue
        candidates = np.asarray(groups[name], dtype=np.int64)
        chunks.append(rng.choice(candidates, size=count, replace=True))
    indices = np.concatenate(chunks)
    rng.shuffle(indices)
    return indices, counts


def build_prompt_strata(model, dataset, prompt_idx, valid_labels, args):
    """Freeze A/B/C/D membership from this prompt's epoch-0 response."""
    sample_count = limited_length(dataset, args.max_train_samples)
    groups = {name: [] for name in ('A', 'B', 'C', 'D')}
    signature = cache_signature(model)
    with torch.no_grad():
        for sample_idx in range(sample_count):
            sample = dataset[sample_idx]
            data_sample = sample['data_samples']
            image_path = data_sample.metainfo['img_path']
            heads = load_heads(
                cache_file(
                    args.prompt_head_cache,
                    'train',
                    prompt_idx,
                    sample_idx,
                    image_path),
                image_path,
                prompt_idx,
                model.query_words[prompt_idx],
                signature,
                model.device)
            target, valid_mask = binary_target(
                data_sample, valid_labels, args.ignore_index, model.device)
            probability = model.calibrate_prompt_heads(prompt_idx, heads)
            gt_pixels = int((target[valid_mask] > 0.5).sum().cpu())
            response_pixels = int((
                probability[valid_mask] >= args.metric_threshold).sum().cpu())
            gt_present = gt_pixels >= args.min_gt_positive_pixels
            response_present = (
                response_pixels >= args.min_response_positive_pixels)
            if gt_present:
                group = 'A' if response_present else 'B'
            else:
                group = 'C' if response_present else 'D'
            groups[group].append(sample_idx)
    return groups


def run_prompt_epoch(model, dataset, split, prompt_idx, valid_labels, args,
                     optimizer=None, epoch=0, sampling_groups=None):
    """Run one cached epoch for exactly one prompt parameter group.

    Passing an optimizer enables training. Without one, the same path performs
    validation under no_grad so score selection matches the training formula.
    """
    training = optimizer is not None
    sample_count = limited_length(
        dataset,
        args.max_train_samples if split == 'train' else args.max_val_samples)
    indices = np.arange(sample_count)
    sampling_counts = None
    if training:
        epoch_seed = args.seed + prompt_idx * 100003 + epoch
        if args.prompt_stratified_sampling:
            if sampling_groups is None:
                raise ValueError(
                    'Stratified training requires fixed prompt groups.')
            epoch_samples = (
                sample_count if args.samples_per_epoch < 0
                else args.samples_per_epoch)
            indices, sampling_counts = stratified_epoch_indices(
                sampling_groups,
                args.prompt_group_ratio,
                epoch_samples,
                epoch_seed)
        else:
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)
    state = empty_metric_state(model)

    for order_idx, sample_idx in enumerate(indices, start=1):
        sample = dataset[int(sample_idx)]
        data_sample = sample['data_samples']
        image_path = data_sample.metainfo['img_path']
        path = cache_file(
            args.prompt_head_cache,
            split,
            prompt_idx,
            int(sample_idx),
            image_path)
        heads = load_heads(
            path,
            image_path,
            prompt_idx,
            model.query_words[prompt_idx],
            cache_signature(model),
            model.device)
        target, valid_mask = binary_target(
            data_sample, valid_labels, args.ignore_index, model.device)

        if training:
            optimizer.zero_grad()
            output = prompt_loss(
                model, prompt_idx, heads, target, valid_mask, args)
        else:
            with torch.no_grad():
                output = prompt_loss(
                    model, prompt_idx, heads, target, valid_mask, args)
        if output is None:
            continue
        loss, probabilities, valid_target, components = output

        gradients = None
        if training:
            loss.backward()
            gradients = {
                name: (
                    float(parameter.grad.detach().abs().cpu())
                    if parameter.grad is not None else 0.0)
                for name, parameter in model.prompt_parameter_group_items(
                    prompt_idx)
            }
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.prompt_parameter_group(prompt_idx),
                    args.grad_clip_norm)
            optimizer.step()
            model.constrain_prompt_parameter_group(
                prompt_idx,
                presence_gain_max=args.presence_gain_max,
                prompt_weight_max=args.prompt_weight_max)

        update_metric_state(
            state,
            float(loss.detach().cpu()),
            probabilities,
            valid_target,
            args.metric_threshold,
            components,
            gradients=gradients)
        if (not training
                and split == 'val'
                and args.val_visualize_every_epochs > 0
                and epoch % args.val_visualize_every_epochs == 0):
            save_validation_visualization(
                model,
                data_sample,
                heads,
                prompt_idx,
                valid_labels,
                epoch,
                args)
        if (args.progress_interval > 0
                and order_idx % args.progress_interval == 0):
            current = finalize_metrics(state)
            print(
                f'[{split}] prompt={prompt_idx} epoch={epoch} '
                f'{order_idx}/{len(indices)} {metrics_text(current)}',
                flush=True)
    metrics = finalize_metrics(state)
    if sampling_counts is not None:
        metrics['sampling_counts'] = sampling_counts
    return metrics


def parameter_values(model, prompt_idx):
    """Serialize one prompt's active internal optimization parameters."""
    return {
        name: float(parameter.detach().cpu())
        for name, parameter in model.prompt_parameter_group_items(prompt_idx)
    }


def exported_parameter_values(model, prompt_idx):
    """Return externally compatible values with prompt_weight in linear space."""
    params = model._bounded_prompt_values()
    result = dict(
        semantic_weight=float(params['semantic_weight'][prompt_idx].detach().cpu()),
        presence_gain=float(params['presence_gain'][prompt_idx].detach().cpu()),
        presence_bias=float(params['presence_bias'][prompt_idx].detach().cpu()),
        prompt_weight=float(params['prompt_weight'][prompt_idx].detach().cpu()),
    )
    if model.prompt_parameterization == 'identifiable_three':
        result['presence_intercept'] = float(
            params['presence_intercept'][prompt_idx].detach().cpu())
    return result


def selection_score(metrics, metric_name):
    """Return the configured validation objective in maximize form."""
    if metric_name == 'loss':
        return -metrics['loss']
    value = metrics.get(metric_name)
    if value is None and metric_name in ('iou', 'hard_iou'):
        return 0.0
    return float(value) if value is not None else -metrics['loss']


def select_best_validation(history, metric_name):
    """Select the highest validation score while preserving the earliest tie."""
    best = None
    best_score = float('-inf')
    for row in history:
        metrics = row.get('validation')
        if metrics is None:
            continue
        score = selection_score(metrics, metric_name)
        if score > best_score + 1e-8:
            best = row
            best_score = score
    if best is None:
        raise ValueError('No validation result is available for selection.')
    return best, best_score


def resolve_identity_fallback(baseline_row, raw_best_row, enabled, min_delta):
    """Apply Stage A identity fallback without discarding raw-best metadata."""
    baseline_iou = baseline_row['validation'].get('hard_iou') or 0.0
    raw_iou = raw_best_row['validation'].get('hard_iou') or 0.0
    gain = raw_iou - baseline_iou
    if not enabled:
        return raw_best_row, False, None
    if gain <= min_delta:
        return baseline_row, True, 'gain_below_or_equal_min_delta'
    if (raw_best_row['validation'].get('recall') or 0.0) <= 0.0:
        return baseline_row, True, 'zero_recall'
    return raw_best_row, False, None


def tensorboard_prompt_name(model, prompt_idx):
    """Build a stable, readable TensorBoard namespace for one prompt."""
    prompt = shared.safe_filename(model.query_words[prompt_idx], 48)
    return f'prompt_{prompt_idx:03d}_{prompt}'


def log_prompt_metrics(writer, model, prompt_idx, epoch, split, metrics):
    """Write losses, overlap metrics, and gradients for one prompt epoch."""
    if writer is None or metrics is None:
        return
    prefix = tensorboard_prompt_name(model, prompt_idx)
    for name in (
            'loss', 'iou', 'miou', 'mean_image_iou', 'soft_iou',
            'hard_iou', 'macc', 'mrecall', 'precision', 'recall', 'f1',
            'predicted_positive_pixels', 'gt_positive_pixels',
            'pred_gt_area_ratio'):
        value = metrics.get(name)
        writer.add_scalar(
            f'{prefix}/{split}/{name}',
            float('nan') if value is None else value,
            epoch)
    for name, value in metrics.get('loss_components', {}).items():
        writer.add_scalar(
            f'{prefix}/{split}/loss_components/{name}', value, epoch)
    if split == 'train':
        for name, value in metrics.get('gradients', {}).items():
            writer.add_scalar(
                f'{prefix}/train/gradients/{name}', value, epoch)


def log_prompt_parameters(writer, model, prompt_idx, epoch):
    """Write parameter values plus signed and absolute initialization deltas."""
    if writer is None:
        return
    prefix = tensorboard_prompt_name(model, prompt_idx)
    initial = model.initial_prompt_parameter_group(prompt_idx)
    for name, parameter in model.prompt_parameter_group_items(prompt_idx):
        value = float(parameter.detach().cpu())
        initial_value = float(initial[name].detach().cpu())
        writer.add_scalar(f'{prefix}/parameters/{name}', value, epoch)
        writer.add_scalar(
            f'{prefix}/parameter_delta/{name}',
            value - initial_value,
            epoch)
        writer.add_scalar(
            f'{prefix}/parameter_delta_abs/{name}',
            abs(value - initial_value),
            epoch)
        if name == 'log_prompt_weight':
            prompt_weight = float(np.exp(value))
            initial_prompt_weight = float(np.exp(initial_value))
            writer.add_scalar(
                f'{prefix}/parameters/prompt_weight', prompt_weight, epoch)
            writer.add_scalar(
                f'{prefix}/parameter_delta/prompt_weight',
                prompt_weight - initial_prompt_weight,
                epoch)
            writer.add_scalar(
                f'{prefix}/parameter_delta_abs/prompt_weight',
                abs(prompt_weight - initial_prompt_weight),
                epoch)
    interpreted = exported_parameter_values(model, prompt_idx)
    for name, value in interpreted.items():
        writer.add_scalar(
            f'{prefix}/interpretable_parameters/{name}', value, epoch)


def prompt_checkpoint_path(model, prompt_idx, save_dir):
    """Return the checkpoint path assigned to one prompt."""
    prompt = shared.safe_filename(model.query_words[prompt_idx], 60)
    return osp.join(
        save_dir, 'prompt_best_params', f'{prompt_idx:03d}_{prompt}.pth')


def train_prompt(model, train_dataset, validation_dataset, prompt_idx, args,
                 writer=None):
    """Optimize and restore the validation-best parameters for one prompt.

    Only this prompt's active leaf parameters enter Adam. A post-training
    tensor comparison enforces that every inactive prompt stayed unchanged.
    """
    checkpoint_path = prompt_checkpoint_path(model, prompt_idx, args.save_dir)
    if args.resume and osp.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if checkpoint.get('prompt_score_version') != model.PROMPT_SCORE_VERSION:
            raise RuntimeError(
                f'Prompt {prompt_idx} checkpoint uses an incompatible score '
                'formula. Rerun without --resume to replace old checkpoints.')
        model.set_prompt_parameter_group(prompt_idx, checkpoint['parameters'])
        print(f'Resumed completed prompt {prompt_idx}: {model.query_words[prompt_idx]}')
        return checkpoint['record']

    for parameter in model.prompt_parameters():
        parameter.requires_grad_(False)
    active_parameters = model.prompt_parameter_group(prompt_idx)
    for parameter in active_parameters:
        parameter.requires_grad_(True)
    trainable_ids = {
        id(parameter) for parameter in model.parameters()
        if parameter.requires_grad}
    if trainable_ids != {id(parameter) for parameter in active_parameters}:
        raise RuntimeError(
            f'Prompt {prompt_idx} optimizer isolation check failed.')

    frozen_before = {
        name: values.detach().cpu().clone()
        for name, values in model.prompt_parameter_items()
    }
    optimizer = torch.optim.Adam(active_parameters, lr=args.lr)
    concept_idx = int(model.query_idx[prompt_idx].item())
    concept = model.concepts[concept_idx]
    valid_labels = concept_valid_labels(
        concept, concept_idx, args.concept_path)
    evaluation_dataset = validation_dataset or train_dataset
    evaluation_split = 'val' if validation_dataset is not None else 'train'

    baseline = run_prompt_epoch(
        model,
        evaluation_dataset,
        evaluation_split,
        prompt_idx,
        valid_labels,
        args,
        epoch=0)
    baseline_parameters = parameter_values(model, prompt_idx)
    baseline_interpretable_parameters = exported_parameter_values(
        model, prompt_idx)
    raw_best_score = selection_score(baseline, args.selection_metric)
    raw_best_epoch = 0
    raw_best_metrics = baseline
    raw_best_parameters = dict(baseline_parameters)
    history = [dict(
        epoch=0,
        train=None,
        validation=baseline,
        optimization_parameters=baseline_parameters,
        parameters=baseline_interpretable_parameters)]
    stale_validations = 0
    sampling_groups = None
    sampling_group_summary = None
    if args.prompt_stratified_sampling:
        sampling_groups = build_prompt_strata(
            model, train_dataset, prompt_idx, valid_labels, args)
        sampling_group_summary = {
            name: len(indices) for name, indices in sampling_groups.items()
        }
        print(
            f'[sampling-groups] prompt={prompt_idx} '
            f'counts={sampling_group_summary} '
            f'ratio={list(args.prompt_group_ratio)}',
            flush=True)
    log_prompt_metrics(
        writer, model, prompt_idx, 0, 'validation', baseline)
    log_prompt_parameters(writer, model, prompt_idx, 0)
    print(
        f'[validation] prompt={prompt_idx} '
        f'name={model.query_words[prompt_idx]!r} epoch=0 '
        f'{metrics_text(baseline)}',
        flush=True)

    for epoch in range(1, args.epochs_per_prompt + 1):
        train_metrics = run_prompt_epoch(
            model,
            train_dataset,
            'train',
            prompt_idx,
            valid_labels,
            args,
            optimizer=optimizer,
            epoch=epoch,
            sampling_groups=sampling_groups)
        log_prompt_metrics(
            writer, model, prompt_idx, epoch, 'train', train_metrics)
        validation_metrics = None
        if epoch % args.validate_every_epochs == 0:
            validation_metrics = run_prompt_epoch(
                model,
                evaluation_dataset,
                evaluation_split,
                prompt_idx,
                valid_labels,
                args,
                epoch=epoch)
            log_prompt_metrics(
                writer,
                model,
                prompt_idx,
                epoch,
                'validation',
                validation_metrics)
            score = selection_score(
                validation_metrics, args.selection_metric)
            if score > raw_best_score + 1e-8:
                raw_best_score = score
                raw_best_epoch = epoch
                raw_best_metrics = validation_metrics
                raw_best_parameters = parameter_values(model, prompt_idx)
                stale_validations = 0
            else:
                stale_validations += 1
        log_prompt_parameters(writer, model, prompt_idx, epoch)
        if writer is not None:
            writer.flush()
        epoch_parameters = parameter_values(model, prompt_idx)
        history.append(dict(
            epoch=epoch,
            train=train_metrics,
            validation=validation_metrics,
            optimization_parameters=epoch_parameters,
            parameters=exported_parameter_values(model, prompt_idx)))
        print(
            f'[prompt] {prompt_idx + 1}/{model.num_queries} '
            f'{model.query_words[prompt_idx]!r} epoch={epoch} '
            f'train=[{metrics_text(train_metrics)}] '
            f'validation=[{metrics_text(validation_metrics)}] '
            f'raw_best_epoch={raw_best_epoch} '
            f'raw_best_score={raw_best_score:.6f}',
            flush=True)
        if (args.patience > 0
                and stale_validations >= args.patience):
            print(f'Early stopping prompt {prompt_idx} at epoch {epoch}.')
            break

    raw_best_row, raw_best_score = select_best_validation(
        history, args.selection_metric)
    raw_best_epoch = raw_best_row['epoch']
    raw_best_metrics = raw_best_row['validation']
    raw_best_parameters = raw_best_row['optimization_parameters']
    selected_row, fallback_applied, fallback_reason = resolve_identity_fallback(
        history[0],
        raw_best_row,
        args.identity_fallback,
        args.identity_fallback_min_delta)
    selected_epoch = selected_row['epoch']
    selected_metrics = selected_row['validation']
    selected_parameters = selected_row['optimization_parameters']

    baseline_hard_iou = baseline.get('hard_iou') or 0.0
    raw_best_hard_iou = raw_best_metrics.get('hard_iou') or 0.0
    raw_gain = raw_best_hard_iou - baseline_hard_iou

    selected_hard_iou = selected_metrics.get('hard_iou') or 0.0
    selected_gain = selected_hard_iou - baseline_hard_iou
    selected_recall = selected_metrics.get('recall') or 0.0
    if (selected_gain > args.identity_fallback_min_delta
            and selected_recall > 0.0):
        prompt_exit = 'prototype'
    elif baseline_hard_iou > args.reject_baseline_iou_threshold:
        prompt_exit = 'identity_fallback'
    else:
        prompt_exit = 'reject'

    model.set_prompt_parameter_group(prompt_idx, selected_parameters)
    exported_selected_parameters = exported_parameter_values(model, prompt_idx)
    if writer is not None:
        for name in (
                'iou', 'miou', 'mean_image_iou', 'soft_iou',
                'macc', 'mrecall', 'precision', 'recall', 'f1'):
            value = selected_metrics.get(name)
            if value is not None:
                writer.add_scalar(
                    f'prompt_summary/best_{name}', value, prompt_idx)
        writer.add_scalar(
            'prompt_summary/selected_epoch', selected_epoch, prompt_idx)
        writer.add_scalar(
            'prompt_summary/raw_best_epoch', raw_best_epoch, prompt_idx)
        writer.add_scalar(
            'prompt_summary/hard_iou_gain_vs_epoch0', selected_gain, prompt_idx)
        for name, value in selected_parameters.items():
            writer.add_scalar(
                f'prompt_summary/best_parameters/{name}',
                value,
                prompt_idx)
        writer.add_scalar(
            'prompt_summary/best_parameters/prompt_weight',
            exported_selected_parameters['prompt_weight'],
            prompt_idx)
    print(
        f'[raw-best] prompt={prompt_idx} name={model.query_words[prompt_idx]!r} '
        f'epoch={raw_best_epoch} {metrics_text(raw_best_metrics)} '
        f'hard_iou_gain={raw_gain:.6f}',
        flush=True)
    print(
        f'[selected] prompt={prompt_idx} '
        f'name={model.query_words[prompt_idx]!r} epoch={selected_epoch} '
        f'{metrics_text(selected_metrics)} exit={prompt_exit} '
        f'fallback={fallback_applied} reason={fallback_reason} '
        f'parameters={exported_selected_parameters} '
        f'optimization_parameters={selected_parameters}',
        flush=True)
    for name, values in model.prompt_parameter_items():
        inactive_mask = torch.ones(model.num_queries, dtype=torch.bool)
        inactive_mask[prompt_idx] = False
        if not torch.equal(
                values.detach().cpu()[inactive_mask],
                frozen_before[name][inactive_mask]):
            raise RuntimeError(
                f'Inactive prompt parameters changed while training '
                f'prompt {prompt_idx}: {name}')

    prompt_key, concept_id, prompt_position = model._prompt_keys()[prompt_idx]
    record = dict(
        experiment_id=args.experiment_id,
        seed=args.seed,
        config_snapshot='config_snapshot.yaml',
        prompt_score_version=model.PROMPT_SCORE_VERSION,
        prompt_parameterization=model.prompt_parameterization,
        prompt_index=prompt_idx,
        prompt=prompt_key,
        prompt_position=prompt_position,
        concept_index=concept_idx,
        concept_id=concept_id,
        concept_name=concept.get('concept_name', concept_id),
        valid_labels=valid_labels,
        selection_metric=args.selection_metric,
        baseline_epoch=0,
        baseline_metrics=baseline,
        baseline_parameters=baseline_interpretable_parameters,
        baseline_optimization_parameters=baseline_parameters,
        raw_best_epoch=raw_best_epoch,
        raw_best_score=raw_best_score,
        raw_best_metrics=raw_best_metrics,
        raw_best_optimization_parameters=raw_best_parameters,
        selected_epoch=selected_epoch,
        selected_metrics=selected_metrics,
        fallback_applied=fallback_applied,
        fallback_reason=fallback_reason,
        prompt_exit=prompt_exit,
        hard_iou_gain_vs_epoch0=selected_gain,
        raw_hard_iou_gain_vs_epoch0=raw_gain,
        best_epoch=selected_epoch,
        best_score=selection_score(selected_metrics, args.selection_metric),
        best_metrics=selected_metrics,
        parameters=exported_selected_parameters,
        optimization_parameters=selected_parameters,
        sampling_group_counts=sampling_group_summary,
        history=history)
    os.makedirs(osp.dirname(checkpoint_path), exist_ok=True)
    torch.save(dict(
        prompt_score_version=model.PROMPT_SCORE_VERSION,
        record=record,
        raw_best_parameters=raw_best_parameters,
        parameters=selected_parameters), checkpoint_path)
    return record


def save_progress(records, model, args):
    """Export prompt results, concept rankings, and diagnose-compatible params."""
    records = sorted(records, key=lambda item: item['prompt_index'])
    for record in records:
        missing = REQUIRED_RESULT_FIELDS - set(record)
        if missing:
            raise ValueError(
                f'Prompt {record.get("prompt_index")} result misses fields: '
                f'{sorted(missing)}')
        if record['prompt_exit'] not in (
                'prototype', 'identity_fallback', 'reject'):
            raise ValueError(
                f'Invalid prompt_exit: {record["prompt_exit"]}')
    with open(osp.join(args.save_dir, 'prompt_optimization_results.json'), 'w',
              encoding='utf-8') as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    with open(osp.join(args.save_dir, 'training_history.json'), 'w',
              encoding='utf-8') as file:
        json.dump({
            str(record['prompt_index']): record['history']
            for record in records
        }, file, ensure_ascii=False, indent=2)

    exits = {
        name: [
            dict(
                prompt_index=record['prompt_index'],
                prompt=record['prompt'],
                concept_id=record['concept_id'],
                hard_iou_gain_vs_epoch0=record['hard_iou_gain_vs_epoch0'])
            for record in records if record['prompt_exit'] == name
        ]
        for name in ('prototype', 'identity_fallback', 'reject')
    }
    exit_summary = dict(
        experiment_id=args.experiment_id,
        seed=args.seed,
        num_prompts=len(records),
        counts={name: len(items) for name, items in exits.items()},
        prompts=exits)
    with open(osp.join(args.save_dir, 'prompt_exit_summary.json'), 'w',
              encoding='utf-8') as file:
        json.dump(exit_summary, file, ensure_ascii=False, indent=2)

    # The batch entry point calls this same function, so both independent
    # trainers now emit the same per_prompt workbook schema as eval.py.
    workbook_prompt_rows = []
    for record in records:
        metrics = record['selected_metrics']
        row = dict(
            Prompt=record['prompt'],
            ConceptID=record['concept_id'],
            ConceptName=record['concept_name'],
            Epoch=record['selected_epoch'],
            Split='validation',
            Threshold=args.metric_threshold,
            IoU=100.0 * metrics.get('iou', float('nan')),
            Acc=100.0 * metrics.get('macc', float('nan')),
            Fscore=100.0 * metrics.get('f1', float('nan')),
            Precision=100.0 * metrics.get('precision', float('nan')),
            Recall=100.0 * metrics.get('recall', float('nan')),
            TP=metrics.get('tp'),
            FP=metrics.get('fp'),
            FN=metrics.get('fn'),
            PredPixels=metrics.get('predicted_positive_pixels'),
            GTPixels=metrics.get('gt_positive_pixels'),
            PredGTAreaRatio=metrics.get('pred_gt_area_ratio'))
        workbook_prompt_rows.append(row)
    MetricsReporter.write_workbook(
        osp.join(args.save_dir, 'results.xlsx'),
        prompt_metrics=workbook_prompt_rows,
        model='SAM3-identifiable-prompts',
        dataset=osp.splitext(osp.basename(args.config))[0])

    concepts = []
    for concept_idx, concept in enumerate(model.concepts):
        candidates = [
            record for record in records
            if record['concept_index'] == concept_idx]
        candidates.sort(
            key=lambda item: (
                item['best_metrics']['iou'] is not None,
                item['best_metrics']['iou'] or -1.0),
            reverse=True)
        concepts.append(dict(
            concept_index=concept_idx,
            concept_id=concept.get('concept_id', str(concept_idx)),
            concept_name=concept.get(
                'concept_name', concept.get('concept_id', str(concept_idx))),
            tnls_dl3=concept.get('tnls_dl3', []),
            prompt_candidates=candidates))
    with open(osp.join(args.save_dir, 'prompt_candidates_by_concept.json'), 'w',
              encoding='utf-8') as file:
        json.dump(concepts, file, ensure_ascii=False, indent=2)

    prompt_params = {
        name: {}
        for name in (
            'semantic_weight',
            'presence_gain',
            'presence_bias',
            'prompt_weight',
        )
    }
    for record in records:
        for name, value in record['parameters'].items():
            if name not in prompt_params:
                continue
            concept_values = prompt_params[name].setdefault(
                record['concept_id'], {})
            concept_values[record['prompt']] = value
    compatible_params = dict(
        version='independent-prompt-stage-a-1.0',
        description=(
            'Best independently trained parameters for each prompt. '
            'Concept aggregation has not been optimized yet.'),
        score_rule=(
            'sigmoid(logit(fused_probability) + presence_gain * '
            '(presence_score - presence_bias) + log(prompt_weight))'),
        prompt_parameterization=model.prompt_parameterization,
        optimization_rule=(
            'raw_semantic_weight, raw_presence_gain and presence_intercept '
            'are trained directly.'
            if model.prompt_parameterization == 'identifiable_three'
            else 'log_prompt_weight is trained directly and '
                 'prompt_weight = exp(log_prompt_weight).'),
        source_config=args.config,
        default=dict(
            confidence_threshold=model.confidence_threshold,
            use_sem_seg=model.use_sem_seg,
            use_transformer_decoder=model.use_transformer_decoder,
            use_presence_score=model.use_presence_score,
            instance_filter_use_presence=model.instance_filter_use_presence,
            fusion_mode=model.fusion_mode,
            semantic_weight=0.5,
            presence_gain=1.0,
            presence_bias=0.5,
            prompt_weight=1.0,
            prompt_threshold=0.0,
            min_prompt_component_area=0.0,
            prompt_fusion_mode=model.prompt_fusion_mode,
            concept_weight=1.0,
            concept_threshold=0.0,
            prob_thd=model.prob_thd),
        prompt_params=prompt_params)
    with open(osp.join(args.save_dir, 'independent_prompt_params.json'), 'w',
              encoding='utf-8') as file:
        json.dump(compatible_params, file, ensure_ascii=False, indent=2)


def save_split_manifest(train_dataset, validation_dataset, args):
    """Record the exact ordered train/validation files used by the run."""
    def describe(dataset, limit):
        if dataset is None:
            return None
        count = limited_length(dataset, limit)
        return dict(
            image_root=osp.abspath(dataset.img_dir),
            annotation_root=osp.abspath(dataset.ann_dir),
            count=count,
            samples=[
                dict(image=osp.abspath(image), annotation=osp.abspath(annotation))
                for image, annotation in dataset.samples[:count]
            ])

    manifest = dict(
        train=describe(train_dataset, args.max_train_samples),
        validation=describe(validation_dataset, args.max_val_samples))
    train_images = {
        item['image'] for item in manifest['train']['samples']
    }
    validation_images = {
        item['image'] for item in (
            manifest['validation']['samples'] if manifest['validation'] else [])
    }
    manifest['train_validation_image_overlap'] = sorted(
        train_images & validation_images)
    with open(osp.join(args.save_dir, 'split_manifest.json'), 'w',
              encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def adjusted_rand_index(labels_a, labels_b):
    """Compute ARI without requiring scikit-learn in the training environment."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError('ARI label vectors must have the same shape.')
    if labels_a.size < 2:
        return None

    def choose_two(value):
        return value * (value - 1) / 2.0

    _, inverse_a = np.unique(labels_a, return_inverse=True)
    _, inverse_b = np.unique(labels_b, return_inverse=True)
    table = np.zeros((inverse_a.max() + 1, inverse_b.max() + 1), dtype=np.int64)
    np.add.at(table, (inverse_a, inverse_b), 1)
    sum_combined = float(sum(choose_two(value) for value in table.flat))
    sum_a = float(sum(choose_two(value) for value in table.sum(axis=1)))
    sum_b = float(sum(choose_two(value) for value in table.sum(axis=0)))
    total_pairs = choose_two(labels_a.size)
    expected = sum_a * sum_b / total_pairs
    maximum = 0.5 * (sum_a + sum_b)
    denominator = maximum - expected
    if abs(denominator) <= 1e-12:
        return 1.0
    return float((sum_combined - expected) / denominator)


def update_cross_seed_cluster_stability(stage_dir, min_ari=0.8,
                                        hard_iou_std_max=0.01):
    """Compare completed seed clusterings on their common prototype prompts."""
    cluster_files = sorted(Path(stage_dir).glob(
        'seed_*/independent_prompt_parameter_clusters.json'))
    clusterings = []
    for path in cluster_files:
        with path.open('r', encoding='utf-8') as file:
            analysis = json.load(file)
        clusterings.append((path.parent.name, {
            int(record['prompt_index']): int(record['cluster'])
            for record in analysis.get('prompts', [])
            if record.get('cluster') is not None
        }))
    pairs = []
    for first_idx in range(len(clusterings)):
        for second_idx in range(first_idx + 1, len(clusterings)):
            first_seed, first = clusterings[first_idx]
            second_seed, second = clusterings[second_idx]
            common = sorted(set(first) & set(second))
            ari = adjusted_rand_index(
                [first[index] for index in common],
                [second[index] for index in common])
            pairs.append(dict(
                first_seed=first_seed,
                second_seed=second_seed,
                common_prototype_prompts=len(common),
                ari=ari))
    valid_ari = [item['ari'] for item in pairs if item['ari'] is not None]
    prompt_metrics = {}
    for result_path in sorted(Path(stage_dir).glob(
            'seed_*/prompt_optimization_results.json')):
        with result_path.open('r', encoding='utf-8') as file:
            records = json.load(file)
        for record in records:
            value = record.get('selected_metrics', {}).get('hard_iou')
            if value is not None:
                prompt_metrics.setdefault(
                    str(record['prompt_index']), []).append(float(value))
    prompt_stds = {
        prompt_index: float(np.std(values))
        for prompt_index, values in prompt_metrics.items()
        if len(values) >= 2
    }
    maximum_prompt_std = max(prompt_stds.values(), default=None)
    mean_ari = float(np.mean(valid_ari)) if valid_ari else None
    stable = (
        len(clusterings) >= 3
        and mean_ari is not None
        and mean_ari >= min_ari
        and maximum_prompt_std is not None
        and maximum_prompt_std <= hard_iou_std_max)
    stability = dict(
        completed_seeds=[name for name, _ in clusterings],
        pairwise=pairs,
        mean_ari=mean_ari,
        prompt_hard_iou_std=prompt_stds,
        maximum_prompt_hard_iou_std=maximum_prompt_std,
        thresholds=dict(
            cluster_min_ari=min_ari,
            prompt_hard_iou_std_max=hard_iou_std_max),
        stable_response_prototypes=stable,
        interpretation=(
            'stable_prototypes' if stable else 'candidate_response_behaviors'))
    with open(Path(stage_dir) / 'cross_seed_cluster_stability.json', 'w',
              encoding='utf-8') as file:
        json.dump(stability, file, ensure_ascii=False, indent=2)
    return stability


def update_stage_a_comparison(args):
    """Regenerate Stage A CSV/Markdown summaries from completed run folders."""
    save_dir = Path(args.save_dir).resolve()
    if not args.experiment_id.startswith('A3-'):
        return
    stage_dir = save_dir.parent
    stage_root = stage_dir.parent
    update_cross_seed_cluster_stability(
        stage_dir,
        min_ari=args.cluster_min_ari,
        hard_iou_std_max=args.prompt_hard_iou_std_max)
    rows = []
    for result_path in sorted(stage_root.glob(
            'A3-*/seed_*/prompt_optimization_results.json')):
        with result_path.open('r', encoding='utf-8') as file:
            records = json.load(file)
        if not records:
            continue
        run_dir = result_path.parent
        cluster_path = run_dir / 'independent_prompt_parameter_clusters.json'
        cluster = {}
        if cluster_path.exists():
            with cluster_path.open('r', encoding='utf-8') as file:
                cluster = json.load(file)
        stability_path = run_dir.parent / 'cross_seed_cluster_stability.json'
        stability = {}
        if stability_path.exists():
            with stability_path.open('r', encoding='utf-8') as file:
                stability = json.load(file)

        def mean_metric(source, name):
            values = [
                record[source].get(name) for record in records
                if record[source].get(name) is not None
            ]
            return float(np.mean(values)) if values else None

        gains = [float(record['hard_iou_gain_vs_epoch0']) for record in records]
        exits = [record['prompt_exit'] for record in records]
        rows.append(dict(
            experiment_id=records[0].get('experiment_id', run_dir.parent.name),
            seed=records[0].get('seed', run_dir.name),
            num_prompts=len(records),
            num_prototype=exits.count('prototype'),
            num_identity_fallback=exits.count('identity_fallback'),
            num_reject=exits.count('reject'),
            mean_epoch0_hard_iou=mean_metric('baseline_metrics', 'hard_iou'),
            mean_selected_hard_iou=mean_metric('selected_metrics', 'hard_iou'),
            mean_hard_iou_gain=float(np.mean(gains)),
            mean_precision=mean_metric('selected_metrics', 'precision'),
            mean_recall=mean_metric('selected_metrics', 'recall'),
            mean_pred_gt_area_ratio=mean_metric(
                'selected_metrics', 'pred_gt_area_ratio'),
            num_prompts_improved=sum(value > 0 for value in gains),
            num_prompts_degraded=sum(value < -1e-12 for value in gains),
            cluster_k=cluster.get('selected_cluster_count'),
            silhouette=cluster.get('silhouette_score'),
            ari_across_seeds=stability.get('mean_ari')))
    if not rows:
        return
    fieldnames = list(rows[0])
    with open(stage_root / 'stage_a_experiment_comparison.csv', 'w',
              newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(stage_root / 'stage_a_experiment_comparison.md', 'w',
              encoding='utf-8') as file:
        file.write('| ' + ' | '.join(fieldnames) + ' |\n')
        file.write('| ' + ' | '.join(['---'] * len(fieldnames)) + ' |\n')
        for row in rows:
            file.write('| ' + ' | '.join(
                '' if row[name] is None else str(row[name])
                for name in fieldnames) + ' |\n')


def run_training(args):
    """Build caches, optimize prompts sequentially, and export stage-one data."""
    cfg = shared.load_config(args.config)
    shared.merge_cfg_options(cfg, args.cfg_options)
    apply_config(args, cfg)
    args.concept_path = shared.get_concept_json_path(cfg, args.config)
    if args.concept_path is None:
        raise ValueError(
            f'Cannot resolve concept_path from config: {args.config}')
    os.makedirs(args.save_dir, exist_ok=True)
    shared.save_experiment_snapshot(args, cfg)
    tensorboard_dir = osp.join(args.save_dir, 'tensorboard')
    writer = shared.build_summary_writer(tensorboard_dir)
    if writer is not None:
        writer.add_text(
            'run/arguments',
            json.dumps(vars(args), ensure_ascii=False, indent=2),
            0)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Stage 0: construct the frozen model and datasets used by every prompt.
    model = shared.build_model(cfg.model)
    model.concepts = load_concepts(args.concept_path)
    if len(model.concepts) != model.num_cls:
        raise ValueError(
            f'Concept count mismatch for {args.concept_path}: '
            f'file defines {len(model.concepts)}, model defines {model.num_cls}.')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_dataset = shared.build_dataset(shared.get_train_dataset_cfg(cfg))
    validation_cfg = shared.get_validation_dataset_cfg(cfg)
    validation_dataset = (
        shared.build_dataset(validation_cfg)
        if validation_cfg is not None else None)
    args.ignore_index = concept_ignore_index(
        args.concept_path, cfg.test_evaluator.get('ignore_index', 255))
    save_split_manifest(train_dataset, validation_dataset, args)

    prompt_indices = (
        list(range(model.num_queries))
        if args.prompt_indices is None
        else list(dict.fromkeys(args.prompt_indices)))
    invalid = [
        index for index in prompt_indices
        if index < 0 or index >= model.num_queries]
    if invalid:
        raise ValueError(f'Invalid prompt indices: {invalid}')

    print(
        f'Independent prompt training: prompts={len(prompt_indices)} '
        f'train_samples={limited_length(train_dataset, args.max_train_samples)} '
        f'val_samples={0 if validation_dataset is None else limited_length(validation_dataset, args.max_val_samples)} '
        f'cache={args.prompt_head_cache} '
        f'selection_metric={args.selection_metric}')
    # Stage 1: build the explicit cache only when it is absent.
    prepare_head_cache(
        model,
        train_dataset,
        validation_dataset,
        prompt_indices,
        args)

    # Stage 2: optimize one isolated parameter group at a time.
    records = []
    for prompt_idx in prompt_indices:
        record = train_prompt(
            model,
            train_dataset,
            validation_dataset,
            prompt_idx,
            args,
            writer=writer)
        records.append(record)
        save_progress(records, model, args)
        if writer is not None:
            writer.flush()

    # Stage 3: save the assembled best parameters for later concept search.
    shared.save_calibration_checkpoint(
        model,
        osp.join(args.save_dir, 'independent_prompt_best_params.pth'),
        epoch=None,
        train_loss=None,
        val_loss=None)
    shared.analyze_prompt_parameter_clusters(
        model,
        osp.join(args.save_dir, 'independent_prompt_parameter_clusters.json'),
        records=records,
        min_hard_iou_gain=args.cluster_min_hard_iou_gain,
        min_cluster_size=args.cluster_min_size,
        max_clusters=args.cluster_max_k)
    update_stage_a_comparison(args)
    if writer is not None:
        writer.flush()
        writer.close()
    print(f'Finished independent optimization for {len(records)} prompts.')


def main():
    """Mirror all run output into both legacy log.txt and Stage A run.log."""
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = osp.join(args.save_dir, 'log.txt')
    run_log_path = osp.join(args.save_dir, 'run.log')
    with (open(log_path, 'a', encoding='utf-8', buffering=1) as log_file,
          open(run_log_path, 'a', encoding='utf-8', buffering=1) as run_log):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file, run_log)
        sys.stderr = TeeStream(original_stderr, log_file, run_log)
        try:
            print(
                f'\n=== independent prompt training started '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")} ===',
                flush=True)
            print(f'log_file={osp.abspath(log_path)}', flush=True)
            run_training(args)
        finally:
            print(
                f'=== independent prompt training ended '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")} ===',
                flush=True)
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == '__main__':
    main()
