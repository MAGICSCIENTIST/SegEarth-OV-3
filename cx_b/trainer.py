"""Two-stage CX-B training and export orchestration."""

from __future__ import annotations

import json
import os
import os.path as osp
import time
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from concept_utils import is_json_path, load_concepts
from metrics_reporting import MetricsReporter, ProgressReporter
import train_learnable_params as shared
import train_prompt_params_independent as original_prompt

from .checkpoints import (
    PROMPT_SCORE_VERSION, atomic_json_save, atomic_torch_save,
    completed_matches, implementation_fingerprint, load_a3_records,
    prompt_identity, sha256_file,
)
from .configuration import config_fingerprint, refiner_settings, effective_joint_train
from .refiners import build_refiner, load_refiner, refiner_payload
from .heads import CachedHeadProvider, OnlineSAMHeadProvider
from .losses import build_binary_loss
from .joint import JointPipeline, JointLoss, map_joint_target, optimize_joint, evaluate_joint
from .joint_initialization import load_joint_initialization
from .reporting import CXBReporter, JointObserver, binary_metrics
from .report_workbook import write_best_workbook
from .modules import (
    A3PromptAdapter, DynamicPromptHead, build_concept_aggregator,
    build_prompt_module,
    load_variant, save_variant,
)


DEFAULT_A3_PARAMETERS = dict(
    semantic_weight=0.5,
    presence_gain=1.0,
    presence_intercept=0.0,
)

METRIC_ALIASES = {
    'IoU': ('IoU', 'hard_iou'),
    'Acc': ('Acc', 'accuracy', 'macc'),
    'Precision': ('Precision', 'precision'),
    'Recall': ('Recall', 'recall'),
    'Fscore': ('Fscore', 'f1'),
}


def should_reuse_a3_prompt(prompt_init, prompt_method, loss_method):
    """Only the explicitly requested exact legacy case may skip training."""
    return (prompt_init == 'a3_reuse'
            and prompt_method == 'A3-2'
            and loss_method == 'A3-2')


def expected_prompt_records(concepts, concept_path, prompt_method='A3-2'):
    records = []
    prompt_index = 0
    for concept_idx, concept in enumerate(concepts):
        prompts = concept.get('prompts', [])
        labels = concept.get('tnls_dl3', [])
        if not labels and not is_json_path(concept_path):
            labels = [concept_idx + 1]  # Original LoveDA entry: 0 is ignore.
        for position, prompt in enumerate(prompts):
            records.append(dict(
                prompt_index=prompt_index,
                prompt=prompt,
                concept_index=concept_idx,
                concept_id=concept.get('concept_id', str(concept_idx)),
                concept_name=(concept.get('concept_name')
                              or concept.get('concept_id') or str(concept_idx)),
                prompt_position=position,
                valid_labels=list(labels),
                prompt_score_version=('segov3_max_presence_v1'
                                      if prompt_method == 'segov3'
                                      else PROMPT_SCORE_VERSION)))
            prompt_index += 1
    return records


def record_parameters(record):
    values = record.get('parameters', {})
    return dict(
        semantic_weight=float(values.get('semantic_weight', 0.5)),
        presence_gain=float(values.get('presence_gain', 1.0)),
        presence_intercept=float(values.get('presence_intercept', 0.0)))


def binary_counts(probability, target, valid_mask, threshold=0.5):
    pred = probability[valid_mask] >= threshold
    gt = target[valid_mask].bool()
    intersect = (pred & gt).sum().item()
    return dict(
        intersect=intersect,
        union=(pred | gt).sum().item(),
        predicted=pred.sum().item(),
        target=gt.sum().item())


def add_counts(total, current):
    for key in ('intersect', 'union', 'predicted', 'target'):
        total[key] = total.get(key, 0) + current[key]


def metrics_log_text(metrics):
    if not metrics:
        return 'metrics=unavailable'
    fields = []
    for key in ('IoU', 'Acc', 'Precision', 'Recall', 'Fscore'):
        value = metrics.get(key)
        fields.append(
            f'{key}={float(value):.6f}' if value is not None
            and np.isfinite(float(value)) else f'{key}=nan')
    fields.extend(
        f'{key}={metrics.get(key)}'
        for key in ('TP', 'FP', 'FN', 'PredPixels', 'GTPixels'))
    return ' '.join(fields)


def metric_value(metrics, metric):
    metrics = metrics or {}
    for key in METRIC_ALIASES[metric]:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def metric_gains(final_metrics, initial_metrics):
    gains = {}
    for metric in METRIC_ALIASES:
        final_value = metric_value(final_metrics, metric)
        initial_value = metric_value(initial_metrics, metric)
        gains[metric] = (
            None if final_value is None or initial_value is None
            else final_value - initial_value)
    return gains


def iou_gain(final_metrics, initial_metrics):
    return metric_gains(final_metrics, initial_metrics)['IoU']


def result_metric_gains(result):
    gains = result.get('metric_gains')
    if gains is None:
        gains = metric_gains(
            result.get('best_metrics'), result.get('initial_metrics'))
    return gains


def evaluate_binary_samples(samples, threshold=0.5):
    counts = {}
    with torch.no_grad():
        for probability, target, valid_mask, *unused in samples():
            add_counts(counts, binary_counts(
                probability, target, valid_mask, threshold))
    return MetricsReporter.counts_to_metrics(
        counts.get('intersect', 0), counts.get('union', 0),
        counts.get('predicted', 0), counts.get('target', 0), percent=False)


def _module_parameters(module):
    return [parameter for parameter in module.parameters()
            if parameter.requires_grad]


def cache_has_missing(cache_root, dataset, split, prompt_indices):
    for sample_idx in range(len(dataset)):
        image_path = dataset[sample_idx]['data_samples'].metainfo['img_path']
        for prompt_idx in prompt_indices:
            if not osp.isfile(original_prompt.cache_file(
                    cache_root, split, prompt_idx, sample_idx, image_path)):
                return True
    return False


def optimize_binary_module(module, train_samples, validation_samples,
                           loss_fn, train_cfg, threshold=0.5,
                           regularization_reference=None, log_prefix=None,
                           checkpoint_callback=None, observer=None):
    """Optimize one independent prompt/concept module.

    Samples are callables returning ``(probability, target, valid_mask)`` so
    cached or online providers can retain their distinct data contracts.
    """
    initial_metrics = evaluate_binary_samples(validation_samples, threshold)
    progress = ProgressReporter(total=int(train_cfg['epochs']))
    if observer:
        observer.validation(0, initial_metrics)
    elif log_prefix:
        print(f'[{log_prefix}] validation epoch=0 '
              f'{metrics_log_text(initial_metrics)} '
              f'{progress.format(0)}', flush=True)
    parameters = _module_parameters(module)
    if not parameters:
        if log_prefix:
            print(f'[{log_prefix}] fixed module; optimization skipped; '
                  'expected_finish=not_applicable',
                  flush=True)
        return dict(
            best_epoch=0, status='fixed', history=[],
            initial_metrics=initial_metrics, best_metrics=initial_metrics,
            metric_gains=metric_gains(initial_metrics, initial_metrics),
            iou_gain_vs_initial=0.0)
    epochs = int(train_cfg['epochs'])
    if initial_metrics['GTPixels'] == 0:
        if log_prefix:
            print(f'[{log_prefix}] no validation-positive pixels; '
                  'optimization skipped.', flush=True)
        return dict(
            best_epoch=0,
            status='untrained_no_validation_positives',
            initial_metrics=initial_metrics,
            best_metrics=initial_metrics,
            metric_gains=metric_gains(initial_metrics, initial_metrics),
            iou_gain_vs_initial=0.0,
            history=[])
    optimizer = torch.optim.Adam(parameters, lr=float(train_cfg['lr']))
    patience = int(train_cfg.get('patience', 0))
    validate_every = int(train_cfg.get('validate_every_epochs', 1))
    checkpoint_interval = int(
        train_cfg.get('checkpoint_interval_epochs', 0))
    save_last_checkpoint = bool(
        train_cfg.get('save_last_checkpoint', True))
    reg_weight = float(train_cfg.get('param_reg_weight', 0.0))
    references = ([value.detach().clone() for value in parameters]
                  if regularization_reference is None
                  else regularization_reference)
    # Epoch 0 is a real validation candidate. This guarantees that ``best``
    # means the maximum validation IoU across the baseline and trained epochs.
    best = dict(
        epoch=0, score=initial_metrics['IoU'],
        state={key: value.detach().cpu().clone()
               for key, value in module.state_dict().items()},
        metrics=initial_metrics)
    stale = 0
    history = []
    completed_epochs = 0
    last_train_loss = None
    last_validation_metrics = None
    for epoch in range(1, epochs + 1):
        completed_epochs = epoch
        module.train()
        losses = []
        train_counts = {}
        last_step = 0
        for step, sample in enumerate(train_samples(), 1):
            last_step = step
            probability, target, valid_mask = sample
            if observer:
                with torch.no_grad():
                    add_counts(train_counts, binary_counts(probability.detach(), target, valid_mask, threshold))
            loss = loss_fn(probability, target, valid_mask)
            if loss is None:
                continue
            if reg_weight:
                loss = loss + reg_weight * sum(
                    (value-reference.to(value)).square().mean()
                    for value, reference in zip(parameters, references))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if observer and step % int(train_cfg.get('progress_interval', 20)) == 0:
                tp = train_counts.get('intersect', 0)
                observer.train(epoch, step, binary_metrics(tp, train_counts.get('predicted', 0)-tp,
                    train_counts.get('target', 0)-tp), float(np.mean(losses)))
        train_loss = float(np.mean(losses)) if losses else None
        if observer:
            tp = train_counts.get('intersect', 0)
            observer.train(epoch, last_step, binary_metrics(tp, train_counts.get('predicted', 0)-tp,
                train_counts.get('target', 0)-tp), train_loss)
        last_train_loss = train_loss
        validation_metrics = None
        should_stop = False
        should_validate = (epoch % validate_every == 0 or epoch == epochs)
        if not should_validate:
            if log_prefix and not observer:
                loss_text = ('nan' if train_loss is None
                             else f'{train_loss:.6f}')
                print(f'[{log_prefix}] epoch={epoch}/{epochs} '
                      f'train_loss={loss_text} validation=skipped '
                      f'{progress.format(epoch)}', flush=True)
        else:
            module.eval()
            validation_metrics = evaluate_binary_samples(
                validation_samples, threshold)
            last_validation_metrics = validation_metrics
            score = validation_metrics['IoU']
            history.append(dict(
                epoch=epoch,
                train_loss=train_loss,
                metrics=validation_metrics))
            if observer:
                observer.validation(epoch, validation_metrics)
            elif log_prefix:
                loss_text = ('nan' if train_loss is None
                             else f'{train_loss:.6f}')
                print(f'[{log_prefix}] epoch={epoch}/{epochs} '
                      f'train_loss={loss_text} validation '
                      f'{metrics_log_text(validation_metrics)} '
                      f'{progress.format(epoch)}', flush=True)
            # Earliest epoch wins ties.
            if best is None or score > best['score'] + 1e-12:
                best = dict(
                    epoch=epoch, score=score,
                    state={key: value.detach().cpu().clone()
                           for key, value in module.state_dict().items()},
                    metrics=validation_metrics)
                stale = 0
            else:
                stale += 1
            should_stop = patience > 0 and stale >= patience
        if (checkpoint_callback is not None and checkpoint_interval > 0
                and epoch % checkpoint_interval == 0):
            checkpoint_callback(
                epoch=epoch, module=module, optimizer=optimizer,
                train_loss=train_loss,
                validation_metrics=validation_metrics,
                checkpoint_type='periodic')
        if should_stop:
            if log_prefix:
                print(f'[{log_prefix}] early_stop epoch={epoch} '
                      f'stale_validations={stale}', flush=True)
            break
    # Save the actual last trained state before restoring the best validation
    # state. ``last.pth`` therefore never silently aliases ``best.pth``.
    if (checkpoint_callback is not None and save_last_checkpoint
            and completed_epochs > 0):
        checkpoint_callback(
            epoch=completed_epochs, module=module, optimizer=optimizer,
            train_loss=last_train_loss,
            validation_metrics=last_validation_metrics,
            checkpoint_type='last')
    if best is not None:
        module.load_state_dict(best['state'])
    gains = metric_gains(
        None if best is None else best['metrics'], initial_metrics)
    result = dict(
        best_epoch=None if best is None else best['epoch'],
        status=('untrained_no_validation_positives'
                if best is not None and best['metrics']['GTPixels'] == 0
                else 'trained'),
        initial_metrics=initial_metrics,
        best_metrics=None if best is None else best['metrics'],
        metric_gains=gains,
        iou_gain_vs_initial=gains['IoU'],
        history=history)
    if log_prefix and not observer:
        print(f'[{log_prefix}] selected epoch={result["best_epoch"]} '
              f'status={result["status"]} '
              f'{metrics_log_text(result["best_metrics"])} '
              f'{progress.format(completed_epochs)}', flush=True)
    return result


class CXBTrainer:
    def __init__(self, config, config_path, output_dir, seed=3407,
                 a3_checkpoint_root=None, selected_concepts=None,
                 resume=True, cli_args=None):
        self.config = config
        self.config_path = osp.abspath(config_path)
        self.cx = config['cx_b']
        self.output_dir = osp.abspath(output_dir)
        self.seed = int(seed)
        self.prompt_init = self.cx.get('prompt_init', 'scratch')
        self.joint_init = self.cx.get('joint_init', 'scratch')
        self.refiner_method, self.refiner_strength = refiner_settings(self.cx)
        if self.refiner_method != 'none' and (self.joint_init != 'checkpoint'
                or not self.cx.get('joint_mode', '').startswith('multiclass_') or selected_concepts is not None):
            raise ValueError('R1 requires a complete multiclass checkpoint and all concepts')
        if self.joint_init not in ('scratch', 'checkpoint'):
            raise ValueError(f'Unknown joint_init: {self.joint_init}')
        if self.joint_init == 'checkpoint':
            if (self.cx.get('joint_mode', 'independent') == 'independent'
                    or not self.cx.get('joint_init_checkpoint') or selected_concepts is not None
                    or self.prompt_init != 'scratch' or a3_checkpoint_root):
                raise ValueError('Joint checkpoint initialization requires a complete joint run, without prompt warmstart')
            source_root = Path(self.cx['joint_init_checkpoint']).resolve().parent.parent
            if Path(self.output_dir).resolve() == source_root or source_root in Path(self.output_dir).resolve().parents:
                raise ValueError('Joint fine-tuning output must be outside the source run')
        self.a3_checkpoint_root = (
            None if a3_checkpoint_root is None
            else osp.abspath(a3_checkpoint_root))
        if self.prompt_init == 'scratch' and self.a3_checkpoint_root:
            raise ValueError(
                'prompt_init=scratch must not receive an A3 checkpoint. '
                'Remove a3_checkpoint_root or choose a3_warmstart/a3_reuse.')
        if self.prompt_init != 'scratch' and not self.a3_checkpoint_root:
            raise ValueError(
                f'prompt_init={self.prompt_init} requires '
                'a3_checkpoint_root.')
        self.selected_concepts = selected_concepts
        self.resume = bool(resume)
        self.report = CXBReporter(self.output_dir, config, cli_args or dict(
            config_path=self.config_path, seed=self.seed, selected_concepts=selected_concepts,
            a3_checkpoint_root=a3_checkpoint_root, resume=resume))
        os.makedirs(self.output_dir, exist_ok=True)

    def _resolved_paths(self):
        concept_path = shared.resolve_config_path(
            self.config.get('concept_path')
            or self.config['model']['classname_path'], self.config_path)
        cache = shared.resolve_config_path(
            self.cx.get('prompt_head_cache'), self.config_path)
        return concept_path, cache

    def run_fingerprint(self):
        concept_path, cache = self._resolved_paths()
        sam_checkpoint = osp.join(
            osp.dirname(osp.dirname(__file__)), 'weights', 'sam3', 'sam3.pt')
        implementation = implementation_fingerprint([
            self.config_path, concept_path, __file__,
            osp.join(osp.dirname(__file__), 'modules.py'),
            osp.join(osp.dirname(__file__), 'losses.py'),
            osp.join(osp.dirname(__file__), 'heads.py'),
            osp.join(osp.dirname(__file__), 'joint.py'),
            osp.join(osp.dirname(__file__), 'reporting.py')])
        if self.cx.get('loss_method') in ('ce_lovasz', 'lovasz'):
            implementation.update(implementation_fingerprint([
                osp.join(osp.dirname(__file__), 'multiclass_lovasz.py')]))
        if self.joint_init == 'checkpoint':
            implementation.update(implementation_fingerprint([
                osp.join(osp.dirname(__file__), 'joint_initialization.py')]))
        if self.refiner_method != 'none':
            implementation.update(implementation_fingerprint([
                osp.join(osp.dirname(__file__), 'refiners.py'),
                osp.join(osp.dirname(__file__), 'configuration.py'),
                osp.join(osp.dirname(__file__), 'multiclass_lovasz.py')]))
        a3_audit = None
        if self.a3_checkpoint_root and osp.isdir(self.a3_checkpoint_root):
            candidate = osp.join(
                self.a3_checkpoint_root, 'prompt_optimization_results.json')
            if osp.isfile(candidate):
                a3_audit = dict(path=candidate, sha256=sha256_file(candidate))
        inputs = dict(
            seed=self.seed,
            prompt_init=self.prompt_init,
            selected_concepts=self.selected_concepts,
            concept_sha256=sha256_file(concept_path),
            concept_content=Path(concept_path).read_text(encoding='utf-8'),
            a3_checkpoint_root=self.a3_checkpoint_root,
            a3_audit=a3_audit,
            prompt_head_cache=cache,
            sam_checkpoint=(
                dict(path=sam_checkpoint, sha256=sha256_file(sam_checkpoint))
                if osp.isfile(sam_checkpoint) else
                dict(path=sam_checkpoint, sha256=None)),
            implementation=implementation)
        if self.joint_init == 'checkpoint':
            source = Path(self.cx['joint_init_checkpoint']).resolve()
            inputs['joint_initialization'] = dict(path=str(source), sha256=sha256_file(source),
                snapshot_sha256=sha256_file(source.parent.parent/'run_config.json'))
        return config_fingerprint(self.config, inputs), inputs

    def _write_run_config(self, fingerprint, inputs):
        atomic_json_save(dict(
            fingerprint=fingerprint,
            config_path=self.config_path,
            config=self.config,
            inputs=inputs,
            seed=self.seed,
            **(dict(train_scope='refiner_only', effective_joint_train=effective_joint_train(self.cx))
               if self.refiner_method != 'none' else {})), osp.join(self.output_dir, 'run_config.json'))

    def _load_base_records(self, expected):
        if self.prompt_init == 'scratch':
            records = []
            for item in expected:
                record = dict(item)
                record.update(
                    parameters=({} if self.cx.get('prompt_method') == 'segov3'
                                else dict(DEFAULT_A3_PARAMETERS)),
                    selected_epoch=None,
                    selected_metrics=None,
                    initialization='scratch')
                records.append(record)
            return records

        if not osp.exists(self.a3_checkpoint_root):
            raise FileNotFoundError(
                f'Explicit A3 source does not exist: '
                f'{self.a3_checkpoint_root}')
        loaded = load_a3_records(
            self.a3_checkpoint_root, expected_records=expected)
        by_key = {prompt_identity(item): item for item in loaded}
        missing = [item for item in expected if prompt_identity(item) not in by_key]
        if missing:
            missing_names = [item['prompt'] for item in missing]
            raise ValueError(
                f'prompt_init={self.prompt_init} requires a complete A3 '
                f'checkpoint; missing prompts: {missing_names}')
        return [by_key[prompt_identity(item)] for item in expected]

    @staticmethod
    def _sample_gt(sample, valid_labels, ignore_index, device):
        return original_prompt.binary_target(
            sample['data_samples'], valid_labels, ignore_index, device)

    @staticmethod
    def _device_heads(heads, device):
        return {key: value.to(device=device, dtype=torch.float32,
                              non_blocking=True)
                for key, value in heads.items()}

    @staticmethod
    def _training_checkpoint_callback(checkpoint_dir, metadata):
        """Create the checkpoint hook consumed by the binary train pipeline."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        def save_checkpoint(*, epoch, module, optimizer, train_loss,
                            validation_metrics, checkpoint_type):
            current_metadata = dict(metadata)
            if isinstance(module, A3PromptAdapter):
                current_metadata['base_parameters'] = module.metadata()
            elif isinstance(module, DynamicPromptHead):
                current_metadata['base_parameters'] = module.base.metadata()
            current_metadata.update(
                checkpoint_type=checkpoint_type,
                epoch=int(epoch),
                train_loss=train_loss,
                validation_metrics=validation_metrics)
            filename = (f'epoch_{int(epoch):04d}.pth'
                        if checkpoint_type == 'periodic' else 'last.pth')
            path = osp.join(checkpoint_dir, filename)
            save_variant(
                path, module, current_metadata,
                training_state=dict(
                    epoch=int(epoch),
                    optimizer_state_dict=optimizer.state_dict(),
                    train_loss=train_loss,
                    validation_metrics=validation_metrics))
            print(f'[checkpoint] type={checkpoint_type} epoch={epoch} '
                  f'path={path}', flush=True)

        return save_checkpoint

    def _prompt_samples(self, module, provider, dataset, split,
                        prompt_record, model):
        def iterator():
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                data_sample = sample['data_samples']
                image_path = data_sample.metainfo['img_path']
                if self.cx['data_mode'] == 'online_sam':
                    output = provider.image_outputs(
                        image_path, [prompt_record['prompt_index']])[
                            prompt_record['prompt_index']]
                    heads = output['heads']
                else:
                    heads = provider.load(
                        split, prompt_record['prompt_index'], sample_idx,
                        image_path)
                    heads = self._device_heads(heads, model.device)
                target, valid_mask = self._sample_gt(
                    sample, prompt_record['valid_labels'],
                    original_prompt.concept_ignore_index(
                        self._resolved_paths()[0], 255), model.device)
                yield module(heads), target, valid_mask
        return iterator

    def _concept_samples(self, aggregator, prompt_modules, provider, dataset,
                         split, members, model, return_metadata=False, metric_observer=None):
        member_indices = [item['prompt_index'] for item in members]
        valid_labels = members[0]['valid_labels']
        ignore_index = original_prompt.concept_ignore_index(
            self._resolved_paths()[0], 255)
        def iterator():
            if metric_observer:
                metric_observer.reset_prompts(split)
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                image_path = sample['data_samples'].metainfo['img_path']
                online = (provider.image_outputs(image_path, member_indices)
                          if self.cx['data_mode'] == 'online_sam' else None)
                probabilities, semantics, instances, presences, regions = (
                    [], [], [], [], [])
                for prompt_idx in member_indices:
                    if online is not None:
                        output = online[prompt_idx]
                        heads = output['heads']
                        regions.append(output['regions'])
                    else:
                        heads = provider.load(
                            split, prompt_idx, sample_idx, image_path)
                        heads = self._device_heads(heads, model.device)
                        regions.append(None)
                    probabilities.append(prompt_modules[prompt_idx](heads))
                    semantics.append(heads['semantic_score_map'])
                    instances.append(heads['instance_score_map'])
                    presences.append(torch.as_tensor(
                        heads['presence_score'], device=model.device).squeeze())
                probabilities = torch.stack(probabilities)
                stacked_heads = dict(
                    semantic_score_map=torch.stack(semantics),
                    instance_score_map=torch.stack(instances))
                presence = torch.stack(presences)
                probability = aggregator(
                    probabilities, presence=presence, heads=stacked_heads,
                    regions=regions)
                target, valid_mask = self._sample_gt(
                    sample, valid_labels, ignore_index, model.device)
                if metric_observer:
                    for index, prompt_idx in enumerate(member_indices):
                        metric_observer.prompt_update(split, prompt_idx, probabilities[index], target, valid_mask)
                if return_metadata:
                    yield probability, target, valid_mask, image_path
                else:
                    yield probability, target, valid_mask
        return iterator

    def _build_runtime(self, concepts, expected, cache_root,
                       include_train=True):
        """Build the frozen upstream model, datasets, and head provider."""
        model = shared.build_model(self.config['model'])
        model.concepts = concepts
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        train_dataset = (
            shared.build_dataset(shared.get_train_dataset_cfg(self.config))
            if include_train else None)
        val_dataset = shared.build_dataset(
            shared.get_validation_dataset_cfg(self.config))
        if self.cx['data_mode'] == 'online_sam':
            provider = OnlineSAMHeadProvider(model, self.cx['sam_region_cfg'])
        else:
            cache_args = SimpleNamespace(
                prompt_head_cache=cache_root,
                max_train_samples=-1,
                max_val_samples=-1,
                rebuild_cache=False,
                cache_dtype=self.cx.get('cache_dtype', 'float16'),
                progress_interval=20)
            all_indices = list(range(len(expected)))
            # Fill only absent entries. Existing damaged or mismatched entries
            # remain untouched and fail identity validation when loaded.
            if (include_train and cache_has_missing(
                    cache_root, train_dataset, 'train', all_indices)):
                original_prompt.build_head_cache(
                    model, train_dataset, 'train', all_indices, cache_args)
            if cache_has_missing(cache_root, val_dataset, 'val', all_indices):
                original_prompt.build_head_cache(
                    model, val_dataset, 'val', all_indices, cache_args)
            provider = CachedHeadProvider(cache_root, model)
        return model, train_dataset, val_dataset, provider

    def _export_concept_probability_files(
            self, aggregator, prompt_modules, provider, val_dataset,
            members, model, concept_idx, concept_id):
        probability_dir = osp.join(
            self.output_dir, 'concept_probabilities', 'val',
            f'concept_{concept_idx:03d}')
        os.makedirs(probability_dir, exist_ok=True)
        export_samples = self._concept_samples(
            aggregator, prompt_modules, provider, val_dataset, 'val',
            members, model, return_metadata=True)
        records = []
        with torch.no_grad():
            for sample_index, (probability, _, _, image_path) in enumerate(
                    export_samples()):
                output_path = osp.join(
                    probability_dir,
                    f'{sample_index:06d}_'
                    f'{shared.safe_filename(Path(image_path).stem)}.npy')
                np.save(
                    output_path, probability.detach().float().cpu().numpy())
                records.append(dict(
                    split='val', concept_index=concept_idx,
                    concept_id=concept_id,
                    sample_index=sample_index, image_path=image_path,
                    probability_path=osp.relpath(
                        output_path, self.output_dir),
                    prompt_indices=[
                        item['prompt_index'] for item in members]))
        return records

    def export_probabilities_only(self):
        """Recompute validation NPY files from completed variant checkpoints."""
        if self.refiner_method != 'none':
            raise ValueError('Use eval_cx_b.py for R1 export; per-concept export would omit the refiner')
        fingerprint, _ = self.run_fingerprint()
        completed_path = osp.join(self.output_dir, 'completed.json')
        if not completed_matches(completed_path, fingerprint):
            raise RuntimeError(
                'Checkpoint-only export requires a completed CX-B run whose '
                'fingerprint matches the current methods, config, seed, and '
                'selected concepts.')
        concept_path, cache_root = self._resolved_paths()
        concepts = load_concepts(concept_path)
        expected = expected_prompt_records(
            concepts, concept_path, self.cx['prompt_method'])
        model, _, val_dataset, provider = self._build_runtime(
            concepts, expected, cache_root, include_train=False)

        prompt_modules = []
        for expected_record in expected:
            checkpoint_path = osp.join(
                self.output_dir,
                ('dynamic_prompt'
                 if self.cx['prompt_method'] == 'dynamic_head'
                 else 'prompt_checkpoints'),
                f"prompt_{expected_record['prompt_index']:03d}", 'best.pth')
            if not osp.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f'Prompt checkpoint is missing: {checkpoint_path}')
            module, metadata = load_variant(
                checkpoint_path, device=model.device)
            saved_identity = metadata.get('identity')
            if (not saved_identity
                    or prompt_identity(saved_identity)
                    != prompt_identity(expected_record)):
                raise ValueError(
                    f'Prompt checkpoint identity mismatch: {checkpoint_path}')
            prompt_modules.append(module)

        selected = [
            index for index in range(len(concepts))
            if self.selected_concepts is None or index in self.selected_concepts
        ]
        progress = ProgressReporter(total=len(selected))
        probability_manifest = []
        for order, concept_idx in enumerate(selected, start=1):
            concept = concepts[concept_idx]
            members = [
                item for item in expected
                if item['concept_index'] == concept_idx
            ]
            concept_dir = osp.join(
                self.output_dir, 'concept',
                f"concept_{concept_idx:03d}_"
                f"{shared.safe_filename(concept['concept_id'])}")
            checkpoint_path = osp.join(concept_dir, 'best.pth')
            if not osp.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f'Concept checkpoint is missing: {checkpoint_path}')
            aggregator, metadata = load_variant(
                checkpoint_path, device=model.device)
            if (metadata.get('method') != self.cx['concept_method']
                    or int(metadata.get('prompt_count', -1)) != len(members)):
                raise ValueError(
                    f'Concept checkpoint metadata mismatch: {checkpoint_path}')
            probability_manifest.extend(
                self._export_concept_probability_files(
                    aggregator, prompt_modules, provider, val_dataset,
                    members, model, concept_idx, concept.get('concept_id')))
            print(
                f'[probability-export] concept={order}/{len(selected)} '
                f'name={concept.get("concept_id")!r} '
                f'{progress.format(order)}', flush=True)
        atomic_json_save(
            probability_manifest,
            osp.join(self.output_dir, 'probability_manifest.json'))
        return dict(
            status='probabilities_exported', fingerprint=fingerprint,
            files=len(probability_manifest))

    def _joint_samples(self, provider, dataset, split, expected, model):
        ignore = original_prompt.concept_ignore_index(self._resolved_paths()[0], 255)
        labels = [next(r['valid_labels'] for r in expected if r['concept_index'] == c)
                  for c in range(max(r['concept_index'] for r in expected)+1)]
        def iterator():
            for index in range(len(dataset)):
                sample = dataset[index]['data_samples']
                image_path = sample.metainfo['img_path']
                online = (provider.image_outputs(image_path, [r['prompt_index'] for r in expected])
                          if self.cx['data_mode'] == 'online_sam' else None)
                heads = {r['prompt_index']: self._device_heads(
                    online[r['prompt_index']]['heads'] if online is not None else
                    provider.load(split, r['prompt_index'], index, image_path), model.device)
                    for r in expected}
                regions = ({i: output['regions'] for i, output in online.items()}
                           if online is not None else None)
                gt = sample.gt_sem_seg.data.squeeze(0).to(model.device)
                target, valid = map_joint_target(gt, labels, ignore)
                yield heads, regions, target, valid
        return iterator

    def _joint_paths(self, concepts, expected):
        root = Path(self.output_dir)
        folder = 'dynamic_prompt' if self.cx['prompt_method'] == 'dynamic_head' else 'prompt_checkpoints'
        prompt_paths = [root/folder/f'prompt_{r["prompt_index"]:03d}'/'best.pth' for r in expected]
        concept_paths = [root/'concept'/f'concept_{i:03d}_{shared.safe_filename(c["concept_id"])}'/'best.pth'
                         for i, c in enumerate(concepts)]
        return prompt_paths, concept_paths

    def _run_joint_stage(self, concepts, expected, model, provider, train_dataset,
                         val_dataset, fingerprint, initialization=None):
        """Refine a two-stage result or a validated whole joint checkpoint."""
        root = Path(self.output_dir)
        self.report.catalog(concepts, expected)
        prompt_paths, concept_paths = self._joint_paths(concepts, expected)
        saved_refiner = None
        if initialization is None:
            loaded = [load_variant(p, model.device) for p in prompt_paths+concept_paths]
        else:
            loaded, provenance = initialization
            saved_refiner = getattr(loaded, 'refiner_payload', None)
            loaded = [(module.to(model.device), metadata) for module, metadata in loaded]
            self.report.line(f'[joint-init] checkpoint={provenance["path"]} '
                f'source_epoch={provenance["source_epoch"]} optimizer=new skipped=prompt,concept')
            atomic_json_save(provenance, root/'joint_initialization.json')
        mode = self.cx['joint_mode']
        refiner = (load_refiner(saved_refiner, model.device) if saved_refiner is not None
                   else build_refiner(self.cx, model.device))
        train_cfg = dict(effective_joint_train(self.cx), binary_threshold=self.cx['metric_threshold'])
        pipeline = JointPipeline([m for m, _ in loaded[:len(expected)]],
                                 [m for m, _ in loaded[len(expected):]], expected, mode,
                                 refiner=refiner, background=train_cfg['background'])
        if refiner is not None:
            self.report.line(f'[refiner] method={refiner.method} strength={refiner.strength} '
                f'train_scope=refiner_only trainable_parameters={sum(p.numel() for p in pipeline.parameters() if p.requires_grad)} '
                f'loss={train_cfg["loss_method"]} binary_weight=0 lr={train_cfg["lr"]}')
        if not 0 <= train_cfg['background'] < len(concepts):
            raise ValueError('joint_train.background is not a valid concept index')
        if initialization is None:
            # Keep the original scratch warmup reports/weights unchanged.
            warmup = root/'warmup'
            warmup.mkdir(exist_ok=True)
            for name in ('prompt_metrics.json', 'concept_metrics.json', 'summary_metrics.json',
                         'resolved_prompt_params.pth'):
                shutil.copy2(root/name, warmup/name)
            for path in prompt_paths+concept_paths:
                destination = warmup/path.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            resolved_path = warmup/'resolved_prompt_params.pth'
        else:
            train_cfg['expected_initial_metrics'] = provenance['source_metrics']
            resolved_path = root/'resolved_prompt_params.pth'
            records = []
            for record, module in zip(expected, pipeline.prompts):
                base = module.base if isinstance(module, DynamicPromptHead) else module
                records.append(dict(record, initialization='joint_checkpoint',
                    parameters=base.metadata() if isinstance(base, A3PromptAdapter) else {}))
            atomic_torch_save(dict(format=('cx_b_resolved_prompt_v1' if self.cx['prompt_method'] == 'segov3'
                else 'cx_b_resolved_a3_v1'), records=records), resolved_path)
            atomic_json_save([dict(identity=prompt_identity(r), prompt_init='joint_checkpoint',
                source=provenance['path'], source_sha256=provenance['sha256'],
                decision='load_joint_and_finetune') for r in records], root/'prompt_checkpoint_manifest.json')

        def save_bundle(name, epoch, current, optimizer, metrics):
            variants = {}
            all_modules = list(current.prompts)+list(current.concepts)
            for path, module, (_, metadata) in zip(prompt_paths+concept_paths, all_modules, loaded):
                meta = dict(metadata, run_fingerprint=fingerprint, epoch=epoch, joint_epoch=epoch, joint_mode=mode,
                            checkpoint_type=name, selection_metric='final_multiclass_mIoU')
                if isinstance(module, A3PromptAdapter):
                    meta['base_parameters'] = module.metadata()
                elif isinstance(module, DynamicPromptHead):
                    meta['base_parameters'] = module.base.metadata()
                    meta['base_trainable'] = any(p.requires_grad for p in module.base.parameters())
                # Warmup's validation_metrics refers to another model; replace it.
                meta['validation_metrics'] = metrics
                variants[path.relative_to(root).as_posix()] = dict(
                    format='cx_b_variant_v1', metadata=meta,
                    state_dict={k: v.detach().cpu().clone() for k, v in module.state_dict().items()})
            bundle = dict(format='cx_b_joint_v1', run_fingerprint=fingerprint,
                mode=mode, epoch=epoch, modules=variants, metrics=metrics,
                decision=dict(threshold=train_cfg['threshold'], background=train_cfg['background']),
                optimizer_state_dict=None if optimizer is None else optimizer.state_dict())
            if current.refiner is not None:
                payload = refiner_payload(current.refiner, run_fingerprint=fingerprint, epoch=epoch,
                    class_names=[c['concept_id'] for c in concepts],
                    prior_methods={k:self.cx[k] for k in ('prompt_method','concept_method')},
                      source_checkpoint=self.cx.get('joint_init_checkpoint'),
                      source_checkpoint_sha256=provenance['sha256'])
                bundle.update(format='cx_b_joint_v2', refiner=payload, train_scope='refiner_only',
                              base_metrics=metrics['base_metrics'])
                if name == 'best':
                    atomic_torch_save(payload, root/'refiner'/'best.pth')
            atomic_torch_save(bundle, root/'joint'/f'{name}.pth')
            if name == 'best':
                for path, payload in variants.items():
                    atomic_torch_save(payload, root/path)

        result = optimize_joint(pipeline,
            self._joint_samples(provider, train_dataset, 'train', expected, model),
            self._joint_samples(provider, val_dataset, 'val', expected, model),
            JointLoss(build_binary_loss(self.config, 'concept'), mode, train_cfg),
            train_cfg, checkpoint_callback=save_bundle,
            observer=JointObserver(self.report, train_cfg['epochs'], len(train_dataset) if train_dataset is not None else None))
        self.report.finish_phase('joint')
        atomic_json_save(result, root/'joint'/'training_metrics.json')
        metrics = result['best_metrics']
        summary = {key: None if metrics[key] is None else 100*metrics[key]
                   for key in ('mIoU', 'mAcc', 'mPrecision', 'mRecall', 'mFscore', 'aAcc')}
        summary.update(best_epoch=result['best_epoch'], InitialmIoU=100*result['initial_metrics']['mIoU'],
                       mIoUGain=100*result['mIoU_gain'])
        atomic_json_save(dict(scale='percent', metric_scope='final_multiclass', metrics=summary,
            joint_mode=mode, best_epoch=result['best_epoch'], decision=dict(
                threshold=train_cfg['threshold'], background=train_cfg['background'])), root/'summary_metrics.json')
        class_rows, concept_rows, prompt_rows = [], [], []
        for c, row in enumerate(metrics['per_class']):
            class_rows.append(dict(Class=concepts[c]['concept_id'], BestEpoch=result['best_epoch'],
                **{key: (None if value is None else value*100) if key in METRIC_ALIASES else value
                   for key, value in row.items()}))
            concept_rows.append(dict(concept_index=c, concept_id=concepts[c]['concept_id'],
                metric_scope='final_multiclass', status='joint_selected', best_epoch=result['best_epoch'],
                metrics=row, initial_metrics=result['initial_metrics']['per_class'][c]))
        for r, row in zip(expected, metrics['prompt_binary']):
            prompt_rows.append(dict(prompt_index=r['prompt_index'], prompt=r['prompt'], concept_id=r['concept_id'],
                metric_scope='independent_binary_diagnostic', status='joint_selected',
                best_epoch=result['best_epoch'], best_metrics=row))
        for stem, rows in [('concept_metrics', concept_rows), ('prompt_metrics', prompt_rows)]:
            atomic_json_save(rows, root/(stem+'.json'))
        resolved = torch.load(resolved_path, map_location='cpu')
        for record, module in zip(resolved['records'], pipeline.prompts):
            base = module.base if isinstance(module, DynamicPromptHead) else module
            record['parameters'] = base.metadata() if isinstance(base, A3PromptAdapter) else {}
            record['selected_epoch'] = result['best_epoch']
            record['selection_metric'] = 'final_multiclass_mIoU'
            record['selected_metrics'] = metrics['prompt_binary'][record['prompt_index']]
            record['status'] = 'joint_selected'
            record['metric_gains_vs_epoch0'] = metric_gains(record['selected_metrics'], record.get('baseline_metrics'))
            record['hard_iou_gain_vs_epoch0'] = record['metric_gains_vs_epoch0']['IoU']
        atomic_torch_save(resolved, root/'resolved_prompt_params.pth')
        atomic_json_save(resolved['records'], root/'prompt_reports'/'prompt_optimization_results.json')
        probability_manifest = []
        if self.cx.get('export_probabilities', False):
            if pipeline.refiner is not None:
                probability_manifest = self._export_refined_probabilities(pipeline, concepts, expected, model, provider, val_dataset)
            else:
                for c, aggregator in enumerate(pipeline.concepts):
                    probability_manifest.extend(self._export_concept_probability_files(
                        aggregator, pipeline.prompts, provider, val_dataset,
                        [r for r in expected if r['concept_index'] == c], model, c, concepts[c]['concept_id']))
        atomic_json_save(probability_manifest, root/'probability_manifest.json')
        return dict(summary=summary, concepts=concept_rows, probability_files=len(probability_manifest), joint=result)

    def _export_refined_probabilities(self, pipeline, concepts, expected, model, provider, val_dataset):
        records = []
        pipeline.eval()
        with torch.no_grad():
            for index, (heads, regions, _, _) in enumerate(self._joint_samples(provider, val_dataset, 'val', expected, model)()):
                scores = pipeline(heads, regions)
                image_path = val_dataset[index]['data_samples'].metainfo['img_path']
                for c, concept in enumerate(concepts):
                    path = Path(self.output_dir)/'concept_probabilities'/'val'/f'concept_{c:03d}'/f'{index:06d}_{shared.safe_filename(Path(image_path).stem)}.npy'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, scores[c].detach().float().cpu().numpy())
                    records.append(dict(split='val', concept_index=c, concept_id=concept['concept_id'],
                        sample_index=index, image_path=str(image_path), probability_path=path.relative_to(self.output_dir).as_posix(),
                        score_scope='refined', prompt_indices=[r['prompt_index'] for r in expected if r['concept_index']==c]))
        return records

    def _final_reports(self, concepts, expected, model, provider, val_dataset, best_epoch):
        """Evaluate restored checkpoints; independent training also reports final IoU."""
        root = Path(self.output_dir)
        self.report.phase('final')
        folder = 'dynamic_prompt' if self.cx['prompt_method'] == 'dynamic_head' else 'prompt_checkpoints'
        prompts = [load_variant(root/folder/f'prompt_{r["prompt_index"]:03d}'/'best.pth', model.device)[0]
                   for r in expected]
        selected = [i for i in range(len(concepts)) if self.selected_concepts is None or i in self.selected_concepts]
        aggregators = {i: load_variant(root/'concept'/f'concept_{i:03d}_{shared.safe_filename(concepts[i]["concept_id"])}'/
                                      'best.pth', model.device)[0] for i in selected}
        if len(selected) == len(concepts):
            refiner = None
            if self.refiner_method != 'none':
                from eval_cx_b import load_run
                _, _, _, _, loaded_prompts, aggregators = load_run(root, model.device)
                prompts = [loaded_prompts[r['prompt_index']] for r in expected]
                refiner = aggregators.refiner
            pipeline = JointPipeline(prompts, [aggregators[i] for i in selected], expected, 'multiclass_concept',
                                     refiner=refiner, background=self.cx.get('prediction_background', 0))
            metrics = evaluate_joint(pipeline, self._joint_samples(provider, val_dataset, 'val', expected, model),
                dict(binary_threshold=self.cx['metric_threshold'], threshold=self.config['model'].get('prob_thd', .1),
                     background=self.cx.get('prediction_background', 65535)))
            tables = self.report.tables(*self.report.mappings(metrics))
            summary = {key: None if metrics[key] is None else metrics[key]*100
                       for key in ('mIoU', 'mAcc', 'mPrecision', 'mRecall', 'mFscore', 'aAcc')}
            scope = 'final_multiclass'
        else:
            prompt_metrics = {r['prompt_index']: evaluate_binary_samples(self._prompt_samples(
                prompts[r['prompt_index']], provider, val_dataset, 'val', r, model), self.cx['metric_threshold'])
                for r in expected}
            concept_metrics = {i: evaluate_binary_samples(self._concept_samples(aggregators[i], prompts, provider,
                val_dataset, 'val', [r for r in expected if r['concept_index']==i], model), self.cx['metric_threshold'])
                for i in selected}
            tables = self.report.tables(prompt_metrics, concept_metrics)
            metrics = dict(prompt_binary=prompt_metrics, concept_binary=concept_metrics, per_class=None)
            summary, scope = dict(mIoU=None, mAcc=None, mPrecision=None, mRecall=None, mFscore=None, aAcc=None), 'partial_concepts'
        summary.update(best_epoch=best_epoch, iou=summary['mIoU'],
                       iou_concept=tables[2][1]['iou'], iou_prompt=tables[2][2]['iou'])
        atomic_json_save(dict(scale='fraction', metric_scope=scope, best_epoch=best_epoch, metrics=metrics), root/'final_metrics.json')
        atomic_json_save(dict(scale='percent', metric_scope=scope, metrics=summary), root/'summary_metrics.json')
        self.report.finish_phase('final')
        self.report.done = True
        self.report.validation(0, 0, tables, final=True, best_epoch=best_epoch)
        write_best_workbook(root)
        return summary

    def run(self):
        with self.report:
            return self._run_training()

    def _run_training(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.cx.get('joint_mode', 'independent') != 'independent' and self.selected_concepts is not None:
            raise ValueError('Joint refinement requires every concept')
        fingerprint, inputs = self.run_fingerprint()
        print(
            f'combination={self.cx["prompt_method"]}/'
            f'{self.cx["concept_method"]}/{self.cx["loss_method"]} '
            f'prompt_init={self.prompt_init} seed={self.seed} '
            f'output_dir={self.output_dir}', flush=True)
        print(
            'checkpoint_intervals '
            f'prompt={self.cx["prompt_train"].get("checkpoint_interval_epochs", 0)} '
            f'concept={self.cx["concept_train"].get("checkpoint_interval_epochs", 0)} '
            f'prompt_save_last={self.cx["prompt_train"].get("save_last_checkpoint", True)} '
            f'concept_save_last={self.cx["concept_train"].get("save_last_checkpoint", True)}',
            flush=True)
        completed_path = osp.join(self.output_dir, 'completed.json')
        required_outputs = [
            'run_config.json', 'resolved_prompt_params.pth',
            'prompt_checkpoint_manifest.json', 'concept_metrics.json',
            'prompt_metrics.json', 'summary_metrics.json',
            'probability_manifest.json',
            'log_fix.txt', 'val_fix.txt', 'final_metrics.json']
        if self.cx.get('joint_mode', 'independent') != 'independent':
            required_outputs += ['joint/best.pth', 'joint/training_metrics.json']
        if self.refiner_method != 'none':
            required_outputs += ['refiner/best.pth']
        if (self.resume and completed_matches(completed_path, fingerprint)
                and all(osp.exists(osp.join(self.output_dir, name))
                        for name in required_outputs)):
            print('Matching completed run found; training skipped.', flush=True)
            if not (Path(self.output_dir)/'result_fix.xlsx').exists():
                write_best_workbook(self.output_dir)
            return dict(status='skipped_complete', fingerprint=fingerprint)
        concept_path, cache_root = self._resolved_paths()
        concepts = load_concepts(concept_path)
        expected = expected_prompt_records(
            concepts, concept_path, self.cx['prompt_method'])
        initialization = None
        if self.joint_init == 'checkpoint':
            prompt_paths, concept_paths = self._joint_paths(concepts, expected)
            slots = [p.relative_to(self.output_dir).as_posix() for p in prompt_paths+concept_paths]
            initialization = load_joint_initialization(self.cx['joint_init_checkpoint'],
                self.config, expected, cache_root, slots)
        self.report.begin_run()
        self._write_run_config(fingerprint, inputs)
        self.report.catalog(concepts, expected)
        if initialization is not None:
            train_cfg = effective_joint_train(self.cx)
            self.report.set_plan(dict(joint=train_cfg['epochs']+1, final=1))
            model, train_dataset, val_dataset, provider = self._build_runtime(
                concepts, expected, cache_root, include_train=train_cfg['epochs'] > 0)
            joint_result = self._run_joint_stage(concepts, expected, model, provider,
                train_dataset, val_dataset, fingerprint, initialization=initialization)
            summary_row = self._final_reports(concepts, expected, model, provider, val_dataset,
                joint_result['joint']['best_epoch'])
            atomic_json_save(dict(fingerprint=fingerprint, completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                status='complete'), completed_path)
            return dict(status='complete', fingerprint=fingerprint, summary=summary_row,
                concepts=joint_result['concepts'], probability_files=joint_result['probability_files'],
                joint_mode=self.cx['joint_mode'], joint_init='checkpoint')
        plan = {f'prompt_{r["prompt_index"]}': self.cx['prompt_train']['epochs']+1 for r in expected}
        plan.update({f'concept_{i}': self.cx['concept_train']['epochs']+1 for i in range(len(concepts))
                     if self.selected_concepts is None or i in self.selected_concepts})
        if self.cx.get('joint_mode', 'independent') != 'independent':
            plan['joint'] = self.cx['joint_train']['epochs']+1
        plan['final'] = 1
        self.report.set_plan(plan)
        base_records = self._load_base_records(expected)
        selected_concept_indices = [
            index for index in range(len(concepts))
            if (self.selected_concepts is None
                or index in self.selected_concepts)
        ]
        overall_progress = ProgressReporter(
            total=len(expected) + len(selected_concept_indices))
        print(f'Loaded concepts={len(concepts)} prompts={len(expected)} '
              f'prompt_init={self.prompt_init} '
              f'a3_source={self.a3_checkpoint_root}', flush=True)
        atomic_torch_save(dict(
            format=('cx_b_resolved_prompt_v1' if self.cx['prompt_method'] == 'segov3'
                    else 'cx_b_resolved_a3_v1'), records=base_records),
            osp.join(self.output_dir, 'resolved_prompt_params.pth'))
        manifest = [dict(
            identity=prompt_identity(record),
            prompt_init=self.prompt_init,
            source=(None if self.prompt_init == 'scratch'
                    else self.a3_checkpoint_root),
            decision=(
                'fixed_segov3' if self.cx['prompt_method'] == 'segov3'
                else 'train_from_scratch' if self.prompt_init == 'scratch'
                else 'reuse_complete_a3' if should_reuse_a3_prompt(
                    self.prompt_init, self.cx['prompt_method'],
                    self.cx['loss_method'])
                else 'warmstart_and_train'))
            for record in base_records]
        atomic_json_save(
            manifest, osp.join(self.output_dir,
                               'prompt_checkpoint_manifest.json'))

        # Heavy dependencies are loaded only after all identity/fingerprint
        # checks pass. The same runtime is reused by checkpoint-only export.
        # The fixed original prompt + mean/max combination has nothing to fit.
        # Keep the existing stages, but avoid generating unused training heads.
        include_train = not (self.cx['prompt_method'] == 'segov3'
                             and self.cx['concept_method'] in ('A3-2', 'max'))
        model, train_dataset, val_dataset, provider = self._build_runtime(
            concepts, expected, cache_root, include_train=include_train)
        # Stage 1: independently train/reuse every prompt, then freeze it.
        prompt_modules = []
        prompt_results = []
        resolved_records = []
        for expected_record, record in zip(expected, base_records):
            observer = self.report.binary('prompt', expected_record['prompt_index'],
                self.cx['prompt_train']['epochs'], len(train_dataset) if train_dataset is not None else None)
            params = ({} if self.cx['prompt_method'] == 'segov3'
                      else record_parameters(record))
            prompt_checkpoint_dir = osp.join(
                self.output_dir,
                ('dynamic_prompt' if self.cx['prompt_method'] == 'dynamic_head'
                 else 'prompt_checkpoints'),
                f"prompt_{expected_record['prompt_index']:03d}")
            prompt_checkpoint_metadata = dict(
                kind='prompt', method=self.cx['prompt_method'],
                base_parameters=params,
                base_trainable=(self.cx['prompt_method'] == 'dynamic_head'
                                and self.prompt_init != 'a3_reuse'),
                hidden_channels=self.cx['prompt_train']['hidden_channels'],
                identity=expected_record, run_fingerprint=fingerprint)
            module = build_prompt_module(
                self.cx['prompt_method'], params,
                hidden_channels=self.cx['prompt_train']['hidden_channels'],
                base_trainable=self.prompt_init != 'a3_reuse',
                trainable=not should_reuse_a3_prompt(
                    self.prompt_init, self.cx['prompt_method'],
                    self.cx['loss_method']))
            module.to(model.device)
            exact_a3_reuse = should_reuse_a3_prompt(
                self.prompt_init, self.cx['prompt_method'],
                self.cx['loss_method'])
            if exact_a3_reuse:
                reused_initial = record.get('baseline_metrics')
                reused_final = record.get('selected_metrics')
                result = dict(
                    best_epoch=record.get('selected_epoch', 0),
                    status='reused_complete_a3',
                    initial_metrics=reused_initial,
                    best_metrics=reused_final,
                    iou_gain_vs_initial=(
                        record.get('hard_iou_gain_vs_epoch0')
                        if record.get('hard_iou_gain_vs_epoch0') is not None
                        else iou_gain(reused_final, reused_initial)),
                    history=[])
                # Re-evaluate this run's validation data; imported historical
                # metrics must not be printed as measurements of this validation.
                observer.validation(0, evaluate_binary_samples(self._prompt_samples(
                    module, provider, val_dataset, 'val', expected_record, model), self.cx['metric_threshold']))
            else:
                prompt_loss = build_binary_loss(self.config, 'prompt')
                train_samples = self._prompt_samples(
                    module, provider, train_dataset, 'train',
                    expected_record, model)
                val_samples = self._prompt_samples(
                    module, provider, val_dataset, 'val', expected_record,
                    model)
                initial_state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()}
                result = optimize_binary_module(
                    module, train_samples, val_samples, prompt_loss,
                    self.cx['prompt_train'],
                    threshold=self.cx['metric_threshold'],
                    log_prefix=(
                        f'prompt {expected_record["prompt_index"] + 1}/'
                        f'{len(expected)} {expected_record["prompt"]!r}'),
                    checkpoint_callback=self._training_checkpoint_callback(
                        prompt_checkpoint_dir,
                        prompt_checkpoint_metadata), observer=observer)
                baseline_metrics = result['initial_metrics']
                if (self.cx['prompt_method'] == 'dynamic_head'
                        and result.get('best_metrics') is not None
                        and result['best_metrics']['GTPixels'] > 0):
                    gain = (result['best_metrics']['IoU']
                            - baseline_metrics['IoU'])
                    if (self.cx['prompt_train'].get('identity_fallback', True)
                            and gain <= self.cx['prompt_train'].get(
                                'identity_fallback_min_delta', 0.005)):
                        module.load_state_dict(initial_state)
                        result.update(
                            status='identity_fallback',
                            best_epoch=0,
                            initial_metrics=baseline_metrics,
                            best_metrics=baseline_metrics,
                            metric_gains=metric_gains(
                                baseline_metrics, baseline_metrics),
                            raw_iou_gain_vs_initial=gain,
                            iou_gain_vs_initial=0.0)
            if isinstance(module, A3PromptAdapter):
                # Export this run's trained values, not only epoch-0 values.
                params = module.metadata()
            elif isinstance(module, DynamicPromptHead):
                params = module.base.metadata()
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            prompt_modules.append(module)
            prompt_checkpoint_metadata['base_parameters'] = params
            save_variant(
                osp.join(prompt_checkpoint_dir, 'best.pth'), module,
                dict(prompt_checkpoint_metadata, checkpoint_type='best',
                     epoch=result.get('best_epoch'),
                     validation_metrics=result.get('best_metrics')))
            gains = result_metric_gains(result)
            prompt_results.append(dict(
                prompt_index=expected_record['prompt_index'],
                prompt=expected_record['prompt'],
                concept_id=expected_record['concept_id'],
                status=result['status'],
                best_epoch=result.get('best_epoch'),
                initial_metrics=result.get('initial_metrics'),
                best_metrics=result.get('best_metrics'),
                metric_gains=gains,
                iou_gain_vs_initial=gains['IoU'],
                accuracy_gain_vs_initial=gains['Acc'],
                precision_gain_vs_initial=gains['Precision'],
                recall_gain_vs_initial=gains['Recall'],
                f1_gain_vs_initial=gains['Fscore'],
                history=result.get('history', [])))
            observer.selected(result.get('best_epoch'), result.get('best_metrics'))
            resolved_record = dict(record)
            resolved_record['parameters'] = params
            resolved_record['initialization'] = self.prompt_init
            resolved_record['selected_epoch'] = result.get('best_epoch')
            resolved_record['baseline_metrics'] = result.get('initial_metrics')
            resolved_record['selected_metrics'] = result.get('best_metrics')
            resolved_record['hard_iou_gain_vs_epoch0'] = result.get(
                'iou_gain_vs_initial')
            resolved_record['metric_gains_vs_epoch0'] = gains
            resolved_record['status'] = result.get('status')
            resolved_records.append(resolved_record)
        atomic_torch_save(dict(
            format=('cx_b_resolved_prompt_v1' if self.cx['prompt_method'] == 'segov3'
                    else 'cx_b_resolved_a3_v1'), records=resolved_records),
            osp.join(self.output_dir, 'resolved_prompt_params.pth'))
        atomic_json_save(prompt_results,
                         osp.join(self.output_dir, 'prompt_metrics.json'))
        prompt_report_dir = osp.join(self.output_dir, 'prompt_reports')
        os.makedirs(prompt_report_dir, exist_ok=True)
        atomic_json_save(
            resolved_records,
            osp.join(prompt_report_dir, 'prompt_optimization_results.json'))

        # Stage 2: optimize each concept independently with prompt modules frozen.
        concept_outputs = []
        probability_manifest = []
        for concept_idx, concept in enumerate(concepts):
            if (self.selected_concepts is not None
                    and concept_idx not in self.selected_concepts):
                continue
            observer = self.report.binary('concept', concept_idx, self.cx['concept_train']['epochs'],
                len(train_dataset) if train_dataset is not None else None)
            members = [item for item in expected
                       if item['concept_index'] == concept_idx]
            aggregator = build_concept_aggregator(
                self.cx['concept_method'], len(members),
                hidden_channels=self.cx['concept_train']['hidden_channels'],
                temperature=self.cx['concept_train']['logsumexp_temperature'])
            aggregator.to(model.device).eval()
            concept_dir = osp.join(
                self.output_dir, 'concept',
                f"concept_{concept_idx:03d}_"
                f"{shared.safe_filename(concept['concept_id'])}")
            concept_checkpoint_metadata = dict(
                kind='concept', method=self.cx['concept_method'],
                prompt_count=len(members), members=members,
                hidden_channels=self.cx['concept_train']['hidden_channels'],
                temperature=self.cx['concept_train']['logsumexp_temperature'],
                run_fingerprint=fingerprint)
            train_samples = self._concept_samples(
                aggregator, prompt_modules, provider, train_dataset, 'train',
                members, model, metric_observer=observer)
            val_samples = self._concept_samples(
                aggregator, prompt_modules, provider, val_dataset, 'val',
                members, model, metric_observer=observer)
            if _module_parameters(aggregator):
                concept_result = optimize_binary_module(
                    aggregator, train_samples, val_samples,
                    build_binary_loss(self.config, 'concept'),
                    self.cx['concept_train'],
                    threshold=self.cx['metric_threshold'],
                    log_prefix=(
                        f'concept {concept_idx + 1}/{len(concepts)} '
                        f'{concept.get("concept_id")!r}'),
                    checkpoint_callback=self._training_checkpoint_callback(
                        concept_dir, concept_checkpoint_metadata), observer=observer)
            else:
                fixed_metrics = evaluate_binary_samples(
                    val_samples, self.cx['metric_threshold'])
                concept_result = dict(
                    best_epoch=0,
                    status='identity' if len(members) == 1 else 'fixed',
                    initial_metrics=fixed_metrics,
                    best_metrics=fixed_metrics,
                    metric_gains=metric_gains(
                        fixed_metrics, fixed_metrics),
                    iou_gain_vs_initial=0.0,
                    history=[])
                observer.validation(0, fixed_metrics)
            aggregator.eval()
            save_variant(
                osp.join(concept_dir, 'best.pth'), aggregator,
                dict(concept_checkpoint_metadata, checkpoint_type='best',
                     epoch=concept_result.get('best_epoch'),
                     validation_metrics=concept_result.get('best_metrics')))
            gains = result_metric_gains(concept_result)
            result_row = dict(
                concept_index=concept_idx,
                concept_id=concept.get('concept_id'),
                prompt_count=len(members),
                status=concept_result['status'],
                best_epoch=concept_result.get('best_epoch'),
                initial_metrics=concept_result.get('initial_metrics'),
                metrics=concept_result.get('best_metrics'),
                metric_gains=gains,
                iou_gain_vs_initial=gains['IoU'],
                accuracy_gain_vs_initial=gains['Acc'],
                precision_gain_vs_initial=gains['Precision'],
                recall_gain_vs_initial=gains['Recall'],
                f1_gain_vs_initial=gains['Fscore'],
                history=concept_result.get('history', []))
            # Report the fixed max baseline without silently replacing the
            # configured aggregator when it performs better.
            max_aggregator = build_concept_aggregator('max', len(members)).to(
                model.device)
            max_samples = self._concept_samples(
                max_aggregator, prompt_modules, provider, val_dataset, 'val',
                members, model)
            result_row['max_baseline'] = evaluate_binary_samples(
                max_samples, self.cx['metric_threshold'])
            learned_iou = (result_row['metrics'] or {}).get('IoU')
            max_iou = result_row['max_baseline'].get('IoU')
            result_row['iou_delta_vs_max'] = (
                None if learned_iou is None or max_iou is None
                else learned_iou - max_iou)
            observer.selected(concept_result.get('best_epoch'), concept_result.get('best_metrics'))
            concept_outputs.append(result_row)
            completed_units = len(prompt_results) + len(concept_outputs)
            print(
                f'[CX-B:overall] {completed_units}/'
                f'{len(expected) + len(selected_concept_indices)} '
                f'{overall_progress.format(completed_units)}', flush=True)

            if (self.cx.get('export_probabilities', False)
                    and self.cx.get('joint_mode', 'independent') == 'independent'):
                probability_manifest.extend(
                    self._export_concept_probability_files(
                        aggregator, prompt_modules, provider, val_dataset,
                        members, model, concept_idx,
                        concept.get('concept_id')))
        atomic_json_save(concept_outputs,
                         osp.join(self.output_dir, 'concept_metrics.json'))
        atomic_json_save(
            probability_manifest,
            osp.join(self.output_dir, 'probability_manifest.json'))
        print(
            f'probability_export='
            f'{"enabled" if self.cx.get("export_probabilities", False) else "disabled"} '
            f'files={len(probability_manifest)}', flush=True)
        summary_row = dict(metric_scope='independent_binary_warmup',
            mIoU=float(np.mean([item['metrics']['IoU'] for item in concept_outputs]))*100
                if concept_outputs else None)
        atomic_json_save(dict(scale='percent', metric_scope='independent_binary_warmup', metrics=summary_row),
                         osp.join(self.output_dir, 'summary_metrics.json'))
        joint_result = None
        if self.cx.get('joint_mode', 'independent') != 'independent':
            joint_result = self._run_joint_stage(concepts, expected, model, provider,
                train_dataset, val_dataset, fingerprint)
            summary_row = joint_result['summary']
            concept_outputs = joint_result['concepts']
        summary_row = self._final_reports(concepts, expected, model, provider, val_dataset,
            'per_module' if joint_result is None else joint_result['joint']['best_epoch'])
        atomic_json_save(dict(
            fingerprint=fingerprint, completed_at=time.strftime(
                '%Y-%m-%d %H:%M:%S'), status='complete'), completed_path)
        return dict(
            status='complete', fingerprint=fingerprint,
            summary=summary_row, concepts=concept_outputs,
            probability_files=(len(probability_manifest) if joint_result is None
                               else joint_result['probability_files']),
            joint_mode=self.cx.get('joint_mode', 'independent'))
