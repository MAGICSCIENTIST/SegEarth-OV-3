"""Joint refinement of frozen SAM heads; no training-entry imports.

Hard validation is argmax + low-confidence background fallback. Competition
training uses a differentiable surrogate; it never backpropagates through argmax.
"""
from __future__ import annotations

import time
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .configuration import JOINT_MODES, validate_pure_lovasz
from .modules import A3PromptAdapter, DynamicPromptHead
from .reporting import LayerCounts
from .multiclass_lovasz import lovasz_softmax


def trainable_prompt(module):
    """A3 reused checkpoints store scalars as buffers; promote them explicitly."""
    base = module.base if isinstance(module, DynamicPromptHead) else module
    if isinstance(base, A3PromptAdapter):
        for name in ('raw_semantic_weight', 'raw_presence_gain', 'presence_intercept'):
            if name in base._buffers:
                value = base._buffers.pop(name)
                base.register_parameter(name, nn.Parameter(value.detach().clone()))
    module.requires_grad_(True)


class JointPipeline(nn.Module):
    def __init__(self, prompts, concepts, records, mode, refiner=None, background=0):
        super().__init__()
        if mode not in JOINT_MODES or mode == 'independent':
            raise ValueError(f'Not a joint mode: {mode}')
        self.prompts = nn.ModuleList(prompts)
        self.concepts = nn.ModuleList(concepts)
        self.records = records
        self.mode = mode
        self.refiner = refiner
        self.background = background
        self.raw_scores = None
        self.members = [[r['prompt_index'] for r in records if r['concept_index'] == c]
                        for c in range(len(concepts))]
        if [r['prompt_index'] for r in records] != list(range(len(prompts))) or not all(self.members):
            raise ValueError('Joint pipeline needs contiguous prompts and every concept')
        for module in self.prompts:
            if mode != 'multiclass_concept':
                trainable_prompt(module)
            else:
                module.requires_grad_(False)
        self.concepts.requires_grad_(mode != 'multiclass_prompt')
        if self.refiner is not None:
            self.prompts.requires_grad_(False)
            self.concepts.requires_grad_(False)
            self.prompts.eval()
            self.concepts.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.refiner is not None:
            self.prompts.eval()
            self.concepts.eval()
        return self

    def forward(self, heads, regions=None, return_prompts=False):
        prompt_scores = [module(heads[i]) for i, module in enumerate(self.prompts)]
        scores = []
        for aggregator, members in zip(self.concepts, self.members):
            scores.append(aggregator(
                torch.stack([prompt_scores[i] for i in members]),
                presence=torch.stack([torch.as_tensor(heads[i]['presence_score'],
                    device=prompt_scores[i].device).squeeze() for i in members]),
                heads={key: torch.stack([heads[i][key] for i in members])
                       for key in ('semantic_score_map', 'instance_score_map')},
                regions=[None if regions is None else regions[i] for i in members]))
        result = torch.stack(scores)
        self.raw_scores = None
        if self.refiner is not None:
            self.raw_scores = result.detach()
            result = self.refiner(result, background=self.background)
        return (result, prompt_scores) if return_prompts else result


def map_joint_target(gt, label_groups, ignore_index):
    """Reject ambiguous/uncovered GT instead of silently dropping valid pixels."""
    gt = gt.long()
    valid = gt != ignore_index
    target = torch.full_like(gt, -1)
    for index, labels in enumerate(label_groups):
        mask = torch.zeros_like(valid)
        for label in labels:
            mask |= gt == label
        mask &= valid
        if (mask & (target >= 0)).any():
            raise ValueError('Overlapping concept GT labels cannot form a multiclass target')
        target[mask] = index
    if (valid & (target < 0)).any():
        raise ValueError('Valid GT labels are not covered by all concepts')
    return target, valid


def final_prediction(scores, threshold=.5, background=0):
    if scores.ndim != 3 or not 0 <= background < scores.shape[0]:
        raise ValueError('Invalid concept scores/background')
    if not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('Expected finite probabilities in [0,1]')
    maximum, prediction = scores.max(dim=0)
    return torch.where(maximum < threshold, background, prediction)


def competition_logits(scores, threshold=.5, background=0, temperature=1.):
    # Flooring the background score at the rejection threshold makes its soft
    # competition reflect the low-confidence fallback. Hard validation retains
    # the exact original < threshold rule, including equality/tie boundaries.
    channels = list(scores.unbind(0))
    channels[background] = torch.maximum(channels[background], scores.new_tensor(threshold))
    return torch.logit(torch.stack(channels).clamp(1e-6, 1-1e-6))/temperature


class JointLoss(nn.Module):
    def __init__(self, binary_loss, mode, config):
        super().__init__()
        self.binary_loss, self.mode, self.config = binary_loss, mode, config
        self.loss_method = config.get('loss_method', 'ce')
        if self.loss_method not in ('ce', 'ce_lovasz', 'lovasz'):
            raise ValueError(f'Unknown joint loss method: {self.loss_method}')
        if self.loss_method in ('ce_lovasz', 'lovasz') and not mode.startswith('multiclass_'):
            raise ValueError(f'{self.loss_method} requires a multiclass joint mode')
        if self.loss_method == 'lovasz':
            validate_pure_lovasz({'competition_weight': 0., 'binary_weight': 0.,
                                  'lovasz_weight': 1., **config})
        self.lovasz_weight = (1. if self.loss_method == 'lovasz' else
                             config.get('lovasz_weight', .25) if self.loss_method == 'ce_lovasz' else 0.)
        if not math.isfinite(self.lovasz_weight) or self.lovasz_weight < 0:
            raise ValueError('lovasz_weight must be finite and non-negative')

    def forward(self, scores, target, valid):
        if not valid.any():
            return None
        if self.loss_method == 'lovasz':
            logits = competition_logits(scores, self.config.get('threshold', .5),
                self.config.get('background', 0), self.config.get('temperature', 1.))
            return lovasz_softmax(logits, target.masked_fill(~valid, -1), ignore_index=-1)
        binary = (torch.stack([self.binary_loss(scores[c], target == c, valid)
                              for c in range(scores.shape[0])]).mean()
                  if self.mode == 'concept_prompt' or self.config.get('binary_weight', .25) != 0
                  else scores.new_zeros(()))
        if self.mode == 'concept_prompt':
            return binary
        logits = competition_logits(scores, self.config.get('threshold', .5),
            self.config.get('background', 0), self.config.get('temperature', 1.))
        competition = F.cross_entropy(logits[:, valid].T, target[valid])
        total = (self.config.get('competition_weight', 1.) * competition
                 + self.config.get('binary_weight', .25) * binary)
        if self.lovasz_weight > 0:
            # Use the SAME competition logits/valid mask as multiclass CE.
            masked_target = target.masked_fill(~valid, -1)
            total = total + self.lovasz_weight * lovasz_softmax(logits, masked_target, ignore_index=-1)
        return total


def confusion_metrics(confusion):
    matrix = np.asarray(confusion, dtype=np.int64)
    tp = matrix.diagonal()
    gt, pred = matrix.sum(1), matrix.sum(0)
    def ratio(a, b):
        return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)
    iou, precision, recall = ratio(tp, gt+pred-tp), ratio(tp, pred), ratio(tp, gt)
    with np.errstate(invalid='ignore', divide='ignore'):
        fscore = 2*precision*recall/(precision+recall)
    def value(v):
        return float(v) if np.isfinite(v) else None
    def mean(v):
        return value(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else None
    rows = [dict(concept_index=c, IoU=value(iou[c]), Acc=value(recall[c]),
                 Precision=value(precision[c]), Recall=value(recall[c]), Fscore=value(fscore[c]),
                 TP=int(tp[c]), FP=int(pred[c]-tp[c]), FN=int(gt[c]-tp[c]),
                 GTPixels=int(gt[c]), PredPixels=int(pred[c])) for c in range(len(tp))]
    return dict(mIoU=mean(iou), mAcc=mean(recall), mPrecision=mean(precision),
                mRecall=mean(recall), mFscore=mean(fscore),
                aAcc=float(tp.sum()/matrix.sum()) if matrix.sum() else None,
                valid_pixels=int(matrix.sum()), per_class=rows, confusion_matrix=matrix.tolist())


def evaluate_joint(pipeline, samples, config):
    pipeline.eval()
    counters = LayerCounts(pipeline.records, len(pipeline.concepts), config.get('binary_threshold', .5),
                           config.get('threshold', .5), config.get('background', 0))
    base_counters = (LayerCounts(pipeline.records, len(pipeline.concepts), config.get('binary_threshold', .5),
                     config.get('threshold', .5), config.get('background', 0)) if pipeline.refiner is not None else None)
    with torch.no_grad():
        for heads, regions, target, valid in samples():
            scores, prompts = pipeline(heads, regions, return_prompts=True)
            counters.update(scores, prompts, target, valid, raw_scores=pipeline.raw_scores)
            if base_counters is not None:
                base_counters.update(pipeline.raw_scores, prompts, target, valid)
    result = counters.result()
    if result['mIoU'] is None:
        raise ValueError('Joint validation has no valid GT pixels')
    if base_counters is not None:
        result['base_metrics'] = base_counters.result()
        result['refiner_mIoU_gain'] = result['mIoU'] - result['base_metrics']['mIoU']
    return result


def optimize_joint(pipeline, train_samples, val_samples, loss_fn, config, checkpoint_callback=None, observer=None):
    initial = evaluate_joint(pipeline, val_samples, config)
    parameters = [p for p in pipeline.parameters() if p.requires_grad]
    best_state = {k: v.detach().cpu().clone() for k, v in pipeline.state_dict().items()}
    best_epoch, best_metrics, stale, history = 0, initial, 0, []
    if observer:
        observer.validation(0, initial)
    else:
        print(f'[joint {pipeline.mode}] epoch=0 final_mIoU={initial["mIoU"]:.6f} '
              f'trainable_parameters={sum(p.numel() for p in parameters)}', flush=True)
    reference = config.get('expected_initial_metrics')
    if reference is not None:
        if (reference.get('mIoU') is None
                or abs(initial['mIoU']-reference['mIoU']) > 1e-5
                or initial.get('valid_pixels') != reference.get('valid_pixels')):
            raise ValueError('Joint initialization epoch 0 does not reproduce source validation '
                f'(mIoU={initial["mIoU"]}, source={reference.get("mIoU")}); no optimizer steps performed')
        print(f'[joint-init-check] epoch=0 source_mIoU={100*reference["mIoU"]:.4f} '
              f'actual_mIoU={100*initial["mIoU"]:.4f} valid_pixels={initial.get("valid_pixels")} passed', flush=True)
    if checkpoint_callback:
        checkpoint_callback('epoch_0000', 0, pipeline, None, initial)
    optimizer = torch.optim.Adam(parameters, lr=config['lr']) if parameters else None
    references = [p.detach().clone() for p in parameters]
    epochs = int(config['epochs']) if parameters else 0
    validate_every = int(config.get('validate_every_epochs', 1))
    interval = int(config.get('checkpoint_interval_epochs', 10))
    started = time.monotonic()
    last_epoch, last_metrics = 0, initial
    for epoch in range(1, epochs+1):
        pipeline.train()
        losses = []
        train_counts = LayerCounts(pipeline.records, len(pipeline.concepts), config.get('binary_threshold', .5),
                                   config.get('threshold', .5), config.get('background', 0)) if observer else None
        last_step = 0
        for step, (heads, regions, target, valid) in enumerate(train_samples(), 1):
            last_step = step
            optimizer.zero_grad(set_to_none=True)
            if observer:
                scores, prompts = pipeline(heads, regions, return_prompts=True)
                train_counts.update(scores, prompts, target, valid, raw_scores=pipeline.raw_scores)
            else:
                scores = pipeline(heads, regions)
            loss = loss_fn(scores, target, valid)
            if loss is None:
                continue
            weight = config.get('param_reg_weight', 0.)
            if weight:
                loss = loss + weight*sum((p-r).square().mean() for p, r in zip(parameters, references))
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite joint loss at epoch {epoch}')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError(f'Nonfinite joint gradient at epoch {epoch}')
            optimizer.step()
            losses.append(float(loss.detach()))
            if observer and step % int(config.get('progress_interval', 20)) == 0:
                observer.train(epoch, step, train_counts.result(), float(np.mean(losses)))
        if not losses:
            raise ValueError('Joint training has no valid supervised samples')
        if observer:
            observer.train(epoch, last_step, train_counts.result(), float(np.mean(losses)))
        last_epoch, last_metrics = epoch, None
        should_stop = False
        if epoch % validate_every == 0 or epoch == epochs:
            last_metrics = evaluate_joint(pipeline, val_samples, config)
            if observer:
                observer.validation(epoch, last_metrics)
            if last_metrics['mIoU'] > best_metrics['mIoU'] + 1e-12:
                best_epoch, best_metrics, stale = epoch, last_metrics, 0
                best_state = {k: v.detach().cpu().clone() for k, v in pipeline.state_dict().items()}
            else:
                stale += 1
            should_stop = config.get('patience', 0) > 0 and stale >= config['patience']
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), metrics=last_metrics))
        if observer:
            observer.reporter.line(f'[selection] selection_metric=iou best_epoch={best_epoch} '
                                  f'best_iou={100*best_metrics["mIoU"]:.4f} stale_validations={stale}')
        else:
            print(f'[joint {pipeline.mode}] epoch={epoch}/{epochs} loss={np.mean(losses):.6f} '
                  f'final_mIoU={None if last_metrics is None else last_metrics["mIoU"]} '
                  f'best_epoch={best_epoch} stale_validations={stale} elapsed={time.monotonic()-started:.0f}s', flush=True)
        if checkpoint_callback and interval > 0 and epoch % interval == 0:
            checkpoint_callback(f'epoch_{epoch:04d}', epoch, pipeline, optimizer, last_metrics)
        if should_stop:
            break
    if checkpoint_callback and config.get('save_last_checkpoint', True):
        checkpoint_callback('last', last_epoch, pipeline, optimizer, last_metrics)
    pipeline.load_state_dict(best_state)
    pipeline.eval()
    if checkpoint_callback:
        checkpoint_callback('best', best_epoch, pipeline, None, best_metrics)
    return dict(status='trained' if parameters and epochs else 'fixed' if not parameters else 'evaluated_only',
                mode=pipeline.mode, best_epoch=best_epoch, completed_epochs=last_epoch,
                selection_metric='final_multiclass_mIoU', scale='fraction',
                initial_metrics=initial, best_metrics=best_metrics, history=history,
                mIoU_gain=best_metrics['mIoU']-initial['mIoU'])
