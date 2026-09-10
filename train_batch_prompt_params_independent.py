"""Train independent prompt calibrations in small prompt batches.

This entry point reuses the single-prompt cache, loss, metrics, and export
format. Prompts in one batch see the same image order but retain independent
parameters, Adam states, validation histories, early stopping, and checkpoints.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
import os.path as osp
import sys
import time

os.environ.setdefault('SEGOV3_LIGHTWEIGHT_TRAIN', '1')

import numpy as np
import torch
import torch.nn.functional as F

from concept_utils import load_concepts
import train_learnable_params as shared
import train_prompt_params_independent as single


def parse_args():
    """Add batch-only options while preserving the original CLI."""
    batch_parser = argparse.ArgumentParser(add_help=False)
    batch_parser.add_argument('--prompt-batch-size', type=int, default=4)
    batch_parser.add_argument(
        '--batch-loss-reduction', choices=['mean', 'sum'], default='mean')
    batch_parser.add_argument('--cache-prefetch-workers', type=int, default=2)
    batch_parser.add_argument('--cache-prefetch-depth', type=int, default=4)
    batch_parser.add_argument(
        '--pin-memory',
        action=argparse.BooleanOptionalAction,
        default=True)
    batch_args, remaining = batch_parser.parse_known_args()

    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        args = single.parse_args()
    finally:
        sys.argv = original_argv

    if batch_args.prompt_batch_size <= 0:
        raise ValueError('--prompt-batch-size must be positive.')
    if batch_args.cache_prefetch_workers < 0:
        raise ValueError('--cache-prefetch-workers must be non-negative.')
    if batch_args.cache_prefetch_depth <= 0:
        raise ValueError('--cache-prefetch-depth must be positive.')
    args.prompt_batch_size = batch_args.prompt_batch_size
    args.batch_loss_reduction = batch_args.batch_loss_reduction
    args.cache_prefetch_workers = batch_args.cache_prefetch_workers
    args.cache_prefetch_depth = batch_args.cache_prefetch_depth
    args.pin_memory = batch_args.pin_memory
    return args


def prompt_batches(prompt_indices, batch_size):
    prompt_indices = list(prompt_indices)
    return [
        prompt_indices[start:start + batch_size]
        for start in range(0, len(prompt_indices), batch_size)
    ]


def reduce_prompt_losses(losses, reduction):
    losses = torch.stack(list(losses))
    if reduction == 'mean':
        return losses.mean()
    if reduction == 'sum':
        return losses.sum()
    raise ValueError(f'Unsupported batch loss reduction: {reduction}')


def apply_prompt_batch_update(model, outputs, optimizers, args):
    """Backpropagate one reduced loss and update each prompt independently."""
    if not outputs:
        return {}

    batch_loss = reduce_prompt_losses(
        (output[0] for output in outputs.values()),
        args.batch_loss_reduction)
    batch_loss.backward()

    gradients = {}
    for prompt_idx in outputs:
        gradients[prompt_idx] = {
            name: (
                parameter.grad.detach().abs()
                if parameter.grad is not None
                else torch.zeros((), device=model.device))
            for name, parameter in model.prompt_parameter_group_items(prompt_idx)
        }
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.prompt_parameter_group(prompt_idx),
                args.grad_clip_norm)
        optimizers[prompt_idx].step()
        model.constrain_prompt_parameter_group(
            prompt_idx,
            presence_gain_max=args.presence_gain_max,
            prompt_weight_max=args.prompt_weight_max)
    return gradients


def stack_prompt_heads(heads_by_prompt, prompt_indices):
    """Stack cached heads for vectorized validation calibration."""
    return {
        name: torch.stack([
            heads_by_prompt[prompt_idx][name]
            for prompt_idx in prompt_indices
        ])
        for name in (
            'instance_score_map', 'semantic_score_map', 'presence_score')
    }


def load_cached_sample(dataset, sample_idx, split, prompt_indices, args,
                       signature, model):
    """Load one dataset sample and its unchanged per-prompt cache files."""
    sample = dataset[int(sample_idx)]
    data_sample = sample['data_samples']
    image_path = data_sample.metainfo['img_path']
    heads_by_prompt = {}
    for prompt_idx in prompt_indices:
        heads_by_prompt[prompt_idx] = single.load_heads_cpu(
            single.cache_file(
                args.prompt_head_cache,
                split,
                prompt_idx,
                int(sample_idx),
                image_path),
            image_path,
            prompt_idx,
            model.query_words[prompt_idx],
            signature)
    return int(sample_idx), data_sample, image_path, heads_by_prompt


def prefetched_cached_samples(dataset, indices, split, prompt_indices, args,
                              signature, model):
    """Yield cache samples in order while a bounded worker pool reads ahead."""
    indices = [int(index) for index in indices]
    if args.cache_prefetch_workers == 0:
        for sample_idx in indices:
            yield load_cached_sample(
                dataset,
                sample_idx,
                split,
                prompt_indices,
                args,
                signature,
                model)
        return

    with ThreadPoolExecutor(
            max_workers=args.cache_prefetch_workers,
            thread_name_prefix='prompt-cache') as executor:
        remaining = iter(indices)
        pending = deque()

        def submit_one():
            try:
                sample_idx = next(remaining)
            except StopIteration:
                return False
            pending.append(executor.submit(
                load_cached_sample,
                dataset,
                sample_idx,
                split,
                prompt_indices,
                args,
                signature,
                model))
            return True

        for _ in range(min(args.cache_prefetch_depth, len(indices))):
            submit_one()
        while pending:
            yield pending.popleft().result()
            submit_one()


def transfer_prompt_heads(heads_by_prompt, prompt_indices, model, args):
    """Stack CPU cache tensors and transfer each head type as one batch."""
    stacked = stack_prompt_heads(heads_by_prompt, prompt_indices)
    use_pinned_memory = (
        args.pin_memory
        and torch.device(model.device).type == 'cuda'
        and torch.cuda.is_available())
    transferred = {}
    for name, value in stacked.items():
        if use_pinned_memory and not value.is_pinned():
            value = value.pin_memory()
        transferred[name] = value.to(
            device=model.device,
            dtype=torch.float32,
            non_blocking=use_pinned_memory)
    return transferred


def prepare_prompt_targets(data_sample, prompt_indices, valid_labels, model,
                           args):
    """Transfer GT once and construct all prompt targets on the GPU."""
    gt_cpu = data_sample.gt_sem_seg.data.squeeze().detach().long()
    if gt_cpu.device.type != 'cpu':
        gt_cpu = gt_cpu.cpu()
    valid_cpu = gt_cpu != args.ignore_index
    if not bool(valid_cpu.any()):
        return None, None

    use_pinned_memory = (
        args.pin_memory
        and torch.device(model.device).type == 'cuda'
        and torch.cuda.is_available())
    if use_pinned_memory and not gt_cpu.is_pinned():
        gt_cpu = gt_cpu.pin_memory()
    gt = gt_cpu.to(device=model.device, non_blocking=use_pinned_memory)
    valid_mask = gt != args.ignore_index
    targets = {}
    for prompt_idx in prompt_indices:
        target = torch.zeros_like(gt, dtype=torch.float32)
        for label in valid_labels[prompt_idx]:
            target = torch.where(
                gt == int(label), torch.ones_like(target), target)
        targets[prompt_idx] = target
    return targets, valid_mask


def empty_device_metric_state(model):
    device = model.device
    return dict(
        loss=torch.zeros((), dtype=torch.float64, device=device),
        samples=torch.zeros((), dtype=torch.int64, device=device),
        tp=torch.zeros((), dtype=torch.int64, device=device),
        fp=torch.zeros((), dtype=torch.int64, device=device),
        fn=torch.zeros((), dtype=torch.int64, device=device),
        tn=torch.zeros((), dtype=torch.int64, device=device),
        positive_pixels=torch.zeros((), dtype=torch.int64, device=device),
        soft_intersection=torch.zeros((), dtype=torch.float64, device=device),
        soft_union=torch.zeros((), dtype=torch.float64, device=device),
        image_iou_sum=torch.zeros((), dtype=torch.float64, device=device),
        image_iou_count=torch.zeros((), dtype=torch.int64, device=device),
        component_sums={
            name: torch.zeros((), dtype=torch.float64, device=device)
            for name in ('bce', 'soft_iou', 'tversky', 'recall',
                         'regularization')
        },
        gradient_sums={
            name: torch.zeros((), dtype=torch.float64, device=device)
            for name in single.model_parameter_names(model)
        },
        gradient_steps=torch.zeros((), dtype=torch.int64, device=device))


def update_device_metric_state(state, loss, probabilities, target, threshold,
                               components, gradients=None):
    """Accumulate metrics on-device without synchronizing every scalar."""
    prediction = probabilities >= threshold
    truth = target > 0.5
    sample_tp = (prediction & truth).sum()
    sample_fp = (prediction & ~truth).sum()
    sample_fn = (~prediction & truth).sum()
    sample_tn = (~prediction & ~truth).sum()
    sample_union = sample_tp + sample_fp + sample_fn

    state['loss'] += loss.detach().to(dtype=torch.float64)
    state['samples'] += 1
    state['tp'] += sample_tp
    state['fp'] += sample_fp
    state['fn'] += sample_fn
    state['tn'] += sample_tn
    state['positive_pixels'] += truth.sum()
    state['soft_intersection'] += (
        probabilities * target).sum().to(dtype=torch.float64)
    state['soft_union'] += (
        probabilities.sum()
        + target.sum()
        - (probabilities * target).sum()).to(dtype=torch.float64)
    state['image_iou_sum'] += torch.where(
        sample_union > 0,
        sample_tp.to(torch.float64) / sample_union.clamp_min(1),
        torch.zeros((), dtype=torch.float64, device=probabilities.device))
    state['image_iou_count'] += (sample_union > 0).to(torch.int64)
    for name, value in components.items():
        state['component_sums'][name] += torch.as_tensor(
            value, device=probabilities.device).detach().to(torch.float64)
    if gradients is not None:
        for name, value in gradients.items():
            state['gradient_sums'][name] += torch.as_tensor(
                value, device=probabilities.device).detach().to(torch.float64)
        state['gradient_steps'] += 1


def finalize_device_metric_states(metric_states):
    """Synchronize all prompt accumulators once at the end of an epoch."""
    prompt_indices = list(metric_states)
    scalar_names = (
        'loss', 'samples', 'tp', 'fp', 'fn', 'tn', 'positive_pixels',
        'soft_intersection', 'soft_union', 'image_iou_sum',
        'image_iou_count', 'gradient_steps')
    component_names = (
        'bce', 'soft_iou', 'tversky', 'recall', 'regularization')
    gradient_names = tuple(next(iter(metric_states.values()))['gradient_sums'])
    scalar_values = torch.stack([
        torch.stack([
            metric_states[prompt_idx][name].to(torch.float64)
            for name in scalar_names
        ])
        for prompt_idx in prompt_indices
    ]).cpu().tolist()
    component_values = torch.stack([
        torch.stack([
            metric_states[prompt_idx]['component_sums'][name]
            for name in component_names
        ])
        for prompt_idx in prompt_indices
    ]).cpu().tolist()
    gradient_values = torch.stack([
        torch.stack([
            metric_states[prompt_idx]['gradient_sums'][name]
            for name in gradient_names
        ])
        for prompt_idx in prompt_indices
    ]).cpu().tolist()

    finalized = {}
    for row_idx, prompt_idx in enumerate(prompt_indices):
        values = dict(zip(scalar_names, scalar_values[row_idx]))
        cpu_state = dict(
            loss=values['loss'],
            samples=int(values['samples']),
            tp=int(values['tp']),
            fp=int(values['fp']),
            fn=int(values['fn']),
            tn=int(values['tn']),
            positive_pixels=int(values['positive_pixels']),
            soft_intersection=values['soft_intersection'],
            soft_union=values['soft_union'],
            image_iou_sum=values['image_iou_sum'],
            image_iou_count=int(values['image_iou_count']),
            component_sums=dict(zip(
                component_names, component_values[row_idx])),
            gradient_sums=dict(zip(
                gradient_names, gradient_values[row_idx])),
            gradient_steps=int(values['gradient_steps']))
        finalized[prompt_idx] = single.finalize_metrics(cpu_state)
    return finalized


def prompt_batch_outputs(model, prompt_indices, logits, targets, valid_masks,
                         args):
    """Compute independent prompt losses with batched pixel operations."""
    target_batch = torch.stack([
        targets[prompt_idx] for prompt_idx in prompt_indices
    ]).to(dtype=logits.dtype)
    valid_batch = torch.stack([
        valid_masks[prompt_idx] for prompt_idx in prompt_indices
    ]).bool()
    valid_float = valid_batch.to(dtype=logits.dtype)
    reduce_dims = tuple(range(1, logits.ndim))
    valid_counts = valid_float.sum(dim=reduce_dims)
    safe_counts = valid_counts.clamp_min(1.0)

    positives = (target_batch * valid_float).sum(dim=reduce_dims)
    negatives = valid_counts - positives
    positive_weights = torch.where(
        positives > 0,
        (negatives / positives.clamp_min(1.0)).clamp(
            min=1.0, max=args.max_positive_weight),
        torch.ones_like(positives))
    weight_shape = (
        (len(prompt_indices),) + (1,) * (logits.ndim - 1))
    bce_pixels = (
        positive_weights.reshape(weight_shape)
        * target_batch * F.softplus(-logits)
        + (1.0 - target_batch) * F.softplus(logits))
    bce_losses = (
        bce_pixels * valid_float).sum(dim=reduce_dims) / safe_counts

    probabilities = torch.sigmoid(logits)
    masked_probabilities = probabilities * valid_float
    intersections = (
        masked_probabilities * target_batch).sum(dim=reduce_dims)
    probability_sums = masked_probabilities.sum(dim=reduce_dims)
    unions = probability_sums + positives - intersections
    soft_iou_losses = torch.where(
        positives > 0,
        1.0 - (intersections + 1e-6) / (unions + 1e-6),
        probability_sums / safe_counts)
    soft_recalls = intersections / positives.clamp_min(1.0)
    recall_losses = torch.where(
        positives > 0,
        torch.relu(args.recall_target - soft_recalls),
        torch.zeros_like(soft_recalls))

    false_positives = (
        masked_probabilities * (1.0 - target_batch)).sum(dim=reduce_dims)
    false_negatives = (
        (1.0 - probabilities) * target_batch * valid_float).sum(
            dim=reduce_dims)
    tversky_scores = (
        (intersections + 1e-6)
        / (intersections
           + args.tversky_alpha_fp * false_positives
           + args.tversky_beta_fn * false_negatives
           + 1e-6))
    tversky_losses = 1.0 - tversky_scores
    metric_losses = (
        tversky_losses if args.loss_name == 'bce_tversky'
        else soft_iou_losses)

    regularizations = []
    for prompt_idx in prompt_indices:
        initial_values = model.initial_prompt_parameter_group(prompt_idx)
        regularizations.append(sum(
            (parameter - initial_values[name]).pow(2)
            for name, parameter in model.prompt_parameter_group_items(
                prompt_idx)))
    regularizations = torch.stack(regularizations)
    total_losses = (
        args.bce_loss_weight * bce_losses
        + args.soft_iou_loss_weight * metric_losses
        + args.recall_loss_weight * recall_losses
        + args.param_reg_weight * regularizations)

    outputs = {}
    for batch_idx, prompt_idx in enumerate(prompt_indices):
        valid_mask = valid_batch[batch_idx]
        components = dict(
            bce=bce_losses[batch_idx].detach(),
            soft_iou=soft_iou_losses[batch_idx].detach(),
            tversky=tversky_losses[batch_idx].detach(),
            recall=recall_losses[batch_idx].detach(),
            regularization=regularizations[batch_idx].detach())
        outputs[prompt_idx] = (
            total_losses[batch_idx],
            probabilities[batch_idx][valid_mask].detach(),
            target_batch[batch_idx][valid_mask].detach(),
            components)
    return outputs


def run_prompt_batch_epoch(model, dataset, split, prompt_indices,
                           valid_labels, args, optimizers=None, epoch=0):
    """Run one epoch for prompts sharing one deterministic image order."""
    prompt_indices = list(prompt_indices)
    training = optimizers is not None
    sample_count = single.limited_length(
        dataset,
        args.max_train_samples if split == 'train' else args.max_val_samples)
    indices = np.arange(sample_count)
    if training:
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(indices)

    metric_states = {
        prompt_idx: empty_device_metric_state(model)
        for prompt_idx in prompt_indices
    }
    signature = single.cache_signature(model)
    started = time.time()

    sample_stream = prefetched_cached_samples(
        dataset,
        indices,
        split,
        prompt_indices,
        args,
        signature,
        model)
    for order_idx, loaded in enumerate(sample_stream, start=1):
        sample_idx, data_sample, image_path, cached_heads = loaded
        batch_heads = transfer_prompt_heads(
            cached_heads, prompt_indices, model, args)
        output_heads = {
            prompt_idx: {
                name: value[batch_idx]
                for name, value in batch_heads.items()
            }
            for batch_idx, prompt_idx in enumerate(prompt_indices)
        }
        targets, valid_mask = prepare_prompt_targets(
            data_sample, prompt_indices, valid_labels, model, args)

        if training:
            for prompt_idx in prompt_indices:
                optimizers[prompt_idx].zero_grad(set_to_none=True)

        outputs = {}
        if targets is not None and training:
            for prompt_idx in prompt_indices:
                output = single.prompt_loss(
                    model,
                    prompt_idx,
                    output_heads[prompt_idx],
                    targets[prompt_idx],
                    valid_mask,
                    args,
                    tensor_components=True)
                if output is not None:
                    outputs[prompt_idx] = output

        if targets is not None and not training and prompt_indices:
            with torch.no_grad():
                batch_logits = model.calibrate_prompt_head_logits_batch(
                    prompt_indices, batch_heads)
                outputs = prompt_batch_outputs(
                    model,
                    prompt_indices,
                    batch_logits,
                    targets,
                    {
                        prompt_idx: valid_mask
                        for prompt_idx in prompt_indices
                    },
                    args)

        gradients = (
            apply_prompt_batch_update(model, outputs, optimizers, args)
            if training else {})

        for prompt_idx, output in outputs.items():
            loss, probabilities, valid_target, components = output
            update_device_metric_state(
                metric_states[prompt_idx],
                loss,
                probabilities,
                valid_target,
                args.metric_threshold,
                components,
                gradients=gradients.get(prompt_idx))
            if (not training
                    and split == 'val'
                    and args.val_visualize_every_epochs > 0
                    and epoch % args.val_visualize_every_epochs == 0):
                single.save_validation_visualization(
                    model,
                    data_sample,
                    output_heads[prompt_idx],
                    prompt_idx,
                    valid_labels[prompt_idx],
                    epoch,
                    args)

        if (args.progress_interval > 0
                and (order_idx % args.progress_interval == 0
                     or order_idx == len(indices))):
            elapsed = time.time() - started
            eta = (
                elapsed / order_idx * (len(indices) - order_idx)
                if order_idx > 0 else 0.0)
            print(
                f'[{split}:prompt-batch] epoch={epoch} '
                f'{order_idx}/{len(indices)} prompts={prompt_indices} '
                f'elapsed={shared.format_duration(elapsed)} '
                f'eta={shared.format_duration(eta)}',
                flush=True)

    return finalize_device_metric_states(metric_states)


def set_batch_trainable(model, prompt_indices):
    for parameter in model.prompt_parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    expected_ids = set()
    for prompt_idx in prompt_indices:
        for parameter in model.prompt_parameter_group(prompt_idx):
            parameter.requires_grad_(True)
            expected_ids.add(id(parameter))

    trainable_ids = {
        id(parameter) for parameter in model.parameters()
        if parameter.requires_grad
    }
    if trainable_ids != expected_ids:
        raise RuntimeError(
            f'Prompt batch optimizer isolation failed for {prompt_indices}.')


def freeze_prompt(model, prompt_idx):
    for parameter in model.prompt_parameter_group(prompt_idx):
        parameter.requires_grad_(False)
        parameter.grad = None


def initialize_prompt_states(model, dataset, split, prompt_indices,
                             valid_labels, args, writer):
    baselines = run_prompt_batch_epoch(
        model,
        dataset,
        split,
        prompt_indices,
        valid_labels,
        args,
        epoch=0)
    states = {}
    for prompt_idx in prompt_indices:
        concept_idx = int(model.query_idx[prompt_idx].item())
        concept = model.concepts[concept_idx]
        baseline = baselines[prompt_idx]
        baseline_parameters = single.parameter_values(model, prompt_idx)
        baseline_interpretable = single.exported_parameter_values(
            model, prompt_idx)
        states[prompt_idx] = dict(
            concept_idx=concept_idx,
            concept=concept,
            valid_labels=valid_labels[prompt_idx],
            baseline=baseline,
            raw_best_score=single.selection_score(
                baseline, args.selection_metric),
            raw_best_epoch=0,
            stale_validations=0,
            history=[dict(
                epoch=0,
                train=None,
                validation=baseline,
                optimization_parameters=baseline_parameters,
                parameters=baseline_interpretable)])
        single.log_prompt_metrics(
            writer, model, prompt_idx, 0, 'validation', baseline)
        single.log_prompt_parameters(writer, model, prompt_idx, 0)
        print(
            f'[validation] prompt={prompt_idx} '
            f'name={model.query_words[prompt_idx]!r} epoch=0 '
            f'{single.metrics_text(baseline)}',
            flush=True)
    return states


def finalize_prompt_state(model, prompt_idx, state, args, writer):
    history = state['history']
    baseline = state['baseline']
    raw_best_row, raw_best_score = single.select_best_validation(
        history, args.selection_metric)
    raw_best_epoch = raw_best_row['epoch']
    raw_best_metrics = raw_best_row['validation']
    raw_best_parameters = raw_best_row['optimization_parameters']
    selected_row, fallback_applied, fallback_reason = (
        single.resolve_identity_fallback(
            history[0],
            raw_best_row,
            args.identity_fallback,
            args.identity_fallback_min_delta))
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
    exported_selected = single.exported_parameter_values(model, prompt_idx)
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
                f'prompt_summary/best_parameters/{name}', value, prompt_idx)
        writer.add_scalar(
            'prompt_summary/best_parameters/prompt_weight',
            exported_selected['prompt_weight'],
            prompt_idx)

    print(
        f'[raw-best] prompt={prompt_idx} '
        f'name={model.query_words[prompt_idx]!r} epoch={raw_best_epoch} '
        f'{single.metrics_text(raw_best_metrics)} hard_iou_gain={raw_gain:.6f}',
        flush=True)
    print(
        f'[selected] prompt={prompt_idx} '
        f'name={model.query_words[prompt_idx]!r} epoch={selected_epoch} '
        f'{single.metrics_text(selected_metrics)} exit={prompt_exit} '
        f'fallback={fallback_applied} reason={fallback_reason} '
        f'parameters={exported_selected} '
        f'optimization_parameters={selected_parameters}',
        flush=True)

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
        concept_index=state['concept_idx'],
        concept_id=concept_id,
        concept_name=state['concept'].get('concept_name', concept_id),
        valid_labels=state['valid_labels'],
        selection_metric=args.selection_metric,
        baseline_epoch=0,
        baseline_metrics=baseline,
        baseline_parameters=history[0]['parameters'],
        baseline_optimization_parameters=history[0][
            'optimization_parameters'],
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
        best_score=single.selection_score(
            selected_metrics, args.selection_metric),
        best_metrics=selected_metrics,
        parameters=exported_selected,
        optimization_parameters=selected_parameters,
        sampling_group_counts=None,
        history=history)

    checkpoint_path = single.prompt_checkpoint_path(
        model, prompt_idx, args.save_dir)
    os.makedirs(osp.dirname(checkpoint_path), exist_ok=True)
    torch.save(dict(
        prompt_score_version=model.PROMPT_SCORE_VERSION,
        record=record,
        raw_best_parameters=raw_best_parameters,
        parameters=selected_parameters), checkpoint_path)
    return record


def train_prompt_batch(model, train_dataset, validation_dataset,
                       prompt_indices, args, writer=None):
    prompt_indices = list(prompt_indices)
    records = []
    pending = []
    for prompt_idx in prompt_indices:
        checkpoint_path = single.prompt_checkpoint_path(
            model, prompt_idx, args.save_dir)
        if args.resume and osp.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if checkpoint.get(
                    'prompt_score_version') != model.PROMPT_SCORE_VERSION:
                raise RuntimeError(
                    f'Prompt {prompt_idx} checkpoint uses an incompatible '
                    'score formula. Rerun without --resume to replace it.')
            model.set_prompt_parameter_group(
                prompt_idx, checkpoint['parameters'])
            records.append(checkpoint['record'])
            print(
                f'Resumed completed prompt {prompt_idx}: '
                f'{model.query_words[prompt_idx]}')
        else:
            pending.append(prompt_idx)
    if not pending:
        return records

    set_batch_trainable(model, pending)
    frozen_before = {
        name: values.detach().cpu().clone()
        for name, values in model.prompt_parameter_items()
    }
    optimizers = {
        prompt_idx: torch.optim.Adam(
            model.prompt_parameter_group(prompt_idx), lr=args.lr)
        for prompt_idx in pending
    }
    valid_labels = {}
    for prompt_idx in pending:
        concept_idx = int(model.query_idx[prompt_idx].item())
        valid_labels[prompt_idx] = single.concept_valid_labels(
            model.concepts[concept_idx], concept_idx, args.concept_path)

    evaluation_dataset = validation_dataset or train_dataset
    evaluation_split = 'val' if validation_dataset is not None else 'train'
    states = initialize_prompt_states(
        model,
        evaluation_dataset,
        evaluation_split,
        pending,
        valid_labels,
        args,
        writer)

    active = list(pending)
    for epoch in range(1, args.epochs_per_prompt + 1):
        train_metrics = run_prompt_batch_epoch(
            model,
            train_dataset,
            'train',
            active,
            valid_labels,
            args,
            optimizers=optimizers,
            epoch=epoch)
        validation_metrics = {}
        if epoch % args.validate_every_epochs == 0:
            validation_metrics = run_prompt_batch_epoch(
                model,
                evaluation_dataset,
                evaluation_split,
                active,
                valid_labels,
                args,
                epoch=epoch)

        stopped = []
        for prompt_idx in active:
            state = states[prompt_idx]
            train_result = train_metrics[prompt_idx]
            validation_result = validation_metrics.get(prompt_idx)
            single.log_prompt_metrics(
                writer, model, prompt_idx, epoch, 'train', train_result)
            if validation_result is not None:
                single.log_prompt_metrics(
                    writer,
                    model,
                    prompt_idx,
                    epoch,
                    'validation',
                    validation_result)
                score = single.selection_score(
                    validation_result, args.selection_metric)
                if score > state['raw_best_score'] + 1e-8:
                    state['raw_best_score'] = score
                    state['raw_best_epoch'] = epoch
                    state['stale_validations'] = 0
                else:
                    state['stale_validations'] += 1

            single.log_prompt_parameters(writer, model, prompt_idx, epoch)
            epoch_parameters = single.parameter_values(model, prompt_idx)
            state['history'].append(dict(
                epoch=epoch,
                train=train_result,
                validation=validation_result,
                optimization_parameters=epoch_parameters,
                parameters=single.exported_parameter_values(
                    model, prompt_idx)))
            print(
                f'[prompt-batch] prompt={prompt_idx + 1}/{model.num_queries} '
                f'{model.query_words[prompt_idx]!r} epoch={epoch} '
                f'train=[{single.metrics_text(train_result)}] '
                f'validation=[{single.metrics_text(validation_result)}] '
                f'raw_best_epoch={state["raw_best_epoch"]} '
                f'raw_best_score={state["raw_best_score"]:.6f}',
                flush=True)
            if (args.patience > 0
                    and state['stale_validations'] >= args.patience):
                stopped.append(prompt_idx)

        for prompt_idx in stopped:
            print(f'Early stopping prompt {prompt_idx} at epoch {epoch}.')
            freeze_prompt(model, prompt_idx)
            active.remove(prompt_idx)
        if writer is not None:
            writer.flush()
        if not active:
            break

    for prompt_idx in pending:
        records.append(finalize_prompt_state(
            model, prompt_idx, states[prompt_idx], args, writer))

    protected_mask = torch.ones(model.num_queries, dtype=torch.bool)
    protected_mask[pending] = False
    for name, values in model.prompt_parameter_items():
        if not torch.equal(
                values.detach().cpu()[protected_mask],
                frozen_before[name][protected_mask]):
            raise RuntimeError(
                f'Parameters outside prompt batch changed: {name}')
    for prompt_idx in pending:
        freeze_prompt(model, prompt_idx)
    return records


def run_training(args):
    """Build caches, then optimize independent prompts in small batches."""
    cfg = shared.load_config(args.config)
    shared.merge_cfg_options(cfg, args.cfg_options)
    single.apply_config(args, cfg)
    if args.prompt_stratified_sampling:
        raise ValueError(
            'Batch prompt training requires --no-prompt-stratified-sampling '
            'so every prompt in a batch can share one image order.')

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
    args.ignore_index = single.concept_ignore_index(
        args.concept_path, cfg.test_evaluator.get('ignore_index', 255))
    single.save_split_manifest(train_dataset, validation_dataset, args)

    prompt_indices = (
        list(range(model.num_queries))
        if args.prompt_indices is None
        else list(dict.fromkeys(args.prompt_indices)))
    invalid = [
        index for index in prompt_indices
        if index < 0 or index >= model.num_queries
    ]
    if invalid:
        raise ValueError(f'Invalid prompt indices: {invalid}')

    batches = prompt_batches(prompt_indices, args.prompt_batch_size)
    print(
        f'Batch independent prompt training: prompts={len(prompt_indices)} '
        f'batches={len(batches)} batch_size={args.prompt_batch_size} '
        f'loss_reduction={args.batch_loss_reduction} '
        f'train_samples={single.limited_length(train_dataset, args.max_train_samples)} '
        f'val_samples={0 if validation_dataset is None else single.limited_length(validation_dataset, args.max_val_samples)} '
        f'cache={args.prompt_head_cache} '
        f'selection_metric={args.selection_metric}')

    single.prepare_head_cache(
        model,
        train_dataset,
        validation_dataset,
        prompt_indices,
        args)

    records = []
    for batch_index, batch_indices in enumerate(batches, start=1):
        print(
            f'=== prompt batch {batch_index}/{len(batches)}: '
            f'{batch_indices} ===',
            flush=True)
        records.extend(train_prompt_batch(
            model,
            train_dataset,
            validation_dataset,
            batch_indices,
            args,
            writer=writer))
        single.save_progress(records, model, args)
        if writer is not None:
            writer.flush()

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
    single.update_stage_a_comparison(args)
    if writer is not None:
        writer.flush()
        writer.close()
    print(f'Finished batch optimization for {len(records)} prompts.')


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = osp.join(args.save_dir, 'log.txt')
    run_log_path = osp.join(args.save_dir, 'run.log')
    with (open(log_path, 'a', encoding='utf-8', buffering=1) as log_file,
          open(run_log_path, 'a', encoding='utf-8', buffering=1) as run_log):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = single.TeeStream(original_stdout, log_file, run_log)
        sys.stderr = single.TeeStream(original_stderr, log_file, run_log)
        try:
            print(
                f'\n=== batch independent prompt training started '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")} ===',
                flush=True)
            print(f'log_file={osp.abspath(log_path)}', flush=True)
            run_training(args)
        finally:
            print(
                f'=== batch independent prompt training ended '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")} ===',
                flush=True)
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == '__main__':
    main()
