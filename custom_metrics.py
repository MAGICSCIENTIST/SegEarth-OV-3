from collections import OrderedDict
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable

from concept_utils import load_concepts
from mmseg.registry import METRICS
from metrics_reporting import MetricsReporter


@METRICS.register_module()
class PromptIoUMetric(BaseMetric):
    """Dataset-level binary metrics for scores before concept aggregation.

    When a concept has no explicit ``tnls_dl3`` labels (for example a plain
    LoveDA prompt TXT), its zero-based concept index is used.  This matches a
    dataset pipeline configured with ``reduce_zero_label=True``.
    """

    supports_output_handlers = False
    supports_class_capture = False

    def __init__(self, concept_path, score_threshold=0.5, ignore_index=255,
                 collect_device='cpu', prefix=None, **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.concepts = load_concepts(concept_path)
        self.score_threshold = float(score_threshold)
        self.ignore_index = int(ignore_index)
        self.prompt_metadata = []
        for concept_idx, concept in enumerate(self.concepts):
            labels = concept.get('tnls_dl3', [])
            valid_labels = ({int(label) for label in labels}
                            if labels else {concept_idx})
            prompts = concept.get('prompts', [])
            if isinstance(prompts, str):
                prompts = [prompts]
            for prompt in prompts:
                self.prompt_metadata.append(dict(
                    prompt=prompt,
                    concept_idx=concept_idx,
                    concept_id=concept.get('concept_id', str(concept_idx)),
                    concept_name=(concept.get('concept_name')
                                  or concept.get('concept_id')
                                  or str(concept_idx)),
                    valid_labels=valid_labels))

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            prompt_scores = data_sample.get(
                'raw_prompt_scores', {}).get('data', None)
            if prompt_scores is None:
                raise RuntimeError(
                    'PromptIoUMetric requires raw_prompt_scores from the model.')
            label = data_sample['gt_sem_seg']['data'].squeeze().to(
                prompt_scores.device)
            valid_mask = label != self.ignore_index
            label_values = label[valid_mask]
            rows = []
            for prompt_idx, metadata in enumerate(self.prompt_metadata):
                pred_mask = (prompt_scores[prompt_idx][valid_mask]
                             > self.score_threshold)
                gt_mask = torch.zeros_like(label_values, dtype=torch.bool)
                for valid_label in metadata['valid_labels']:
                    gt_mask |= label_values == valid_label
                intersect = (pred_mask & gt_mask).sum()
                union = (pred_mask | gt_mask).sum()
                rows.append(torch.stack((
                    intersect, union, pred_mask.sum(), gt_mask.sum())).cpu())
            self.results.append(torch.stack(rows))

    def compute_metrics(self, results):
        totals = torch.stack(results).sum(dim=0)
        self._last_prompt_metrics = []
        for prompt_idx, metadata in enumerate(self.prompt_metadata):
            intersect, union, predicted, target = totals[prompt_idx].tolist()
            row = dict(
                Prompt=metadata['prompt'],
                ConceptID=metadata['concept_id'],
                ConceptName=metadata['concept_name'],
                Threshold=self.score_threshold)
            row.update(MetricsReporter.counts_to_metrics(
                intersect, union, predicted, target, percent=True))
            self._last_prompt_metrics.append(row)
        # This auxiliary evaluator must not modify the legacy summary metrics.
        return {}


@METRICS.register_module()
class TNLSConceptIoUMetric(BaseMetric):
    """Evaluate concept predictions against TNLS DL3 label lists."""

    def __init__(self,
                 concept_path: str,
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 keep_results: bool = True,
                 vis_dir: Optional[str] = None,
                 gt_alpha: float = 0.55,
                 score_threshold: float = 0.1,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.keep_results = keep_results
        self.vis_dir = vis_dir
        self.gt_alpha = gt_alpha
        self.score_threshold = score_threshold
        self.concepts = load_concepts(concept_path)
        self.class_names = [
            concept.get('concept_name') or concept.get('concept_id')
            for concept in self.concepts
        ]
        self.label_sets = [
            set(int(label) for label in concept.get('tnls_dl3', []))
            for concept in self.concepts
        ]
        self.prompt_metadata = []
        for concept_idx, concept in enumerate(self.concepts):
            prompts = concept.get('prompts', [])
            if isinstance(prompts, str):
                prompts = [prompts]
            for prompt in prompts:
                self.prompt_metadata.append(
                    dict(
                        prompt=prompt,
                        concept_idx=concept_idx,
                        concept_id=concept.get('concept_id'),
                        concept_name=concept.get('concept_name')
                        or concept.get('concept_id')))

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            class_scores = data_sample.get('seg_logits',
                                           {}).get('data', None)
            prompt_scores = data_sample.get('raw_prompt_scores',
                                            {}).get('data', None)
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            self.results.append(
                self.intersect_and_union(
                    pred_label,
                    label,
                    self.label_sets,
                    self.ignore_index,
                    class_scores=class_scores,
                    prompt_scores=prompt_scores,
                    prompt_metadata=self.prompt_metadata,
                    score_threshold=self.score_threshold))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        results = tuple(zip(*results))
        assert len(results) == 8

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        prompt_area_intersect = sum(results[4])
        prompt_area_union = sum(results[5])
        prompt_area_pred_label = sum(results[6])
        prompt_area_label = sum(results[7])

        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.nan_to_num, self.beta)

        metrics = OrderedDict()
        for key, value in ret_metrics.items():
            if key == 'aAcc':
                metrics[key] = np.round(value * 100, 2)
            else:
                metrics['m' + key] = np.round(np.nanmean(value) * 100, 2)

        class_metrics = OrderedDict({
            key: np.round(value * 100, 2)
            for key, value in ret_metrics.items()
            if key != 'aAcc'
        })
        class_metrics.update({'Class': self.class_names})
        class_metrics.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, value in class_metrics.items():
            class_table_data.add_column(key, value)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        prompt_metrics = self.total_area_to_metrics(
            prompt_area_intersect, prompt_area_union,
            prompt_area_pred_label, prompt_area_label,
            self.nan_to_num, self.beta)
        prompt_metrics.pop('aAcc', None)
        self._last_prompt_metrics = []
        for prompt_idx, metadata in enumerate(self.prompt_metadata):
            self._last_prompt_metrics.append(
                dict(
                    Prompt=metadata['prompt'],
                    ConceptID=metadata['concept_id'],
                    ConceptName=metadata['concept_name'],
                    IoU=self.format_metric(prompt_metrics['IoU'][prompt_idx]),
                    Acc=self.format_metric(prompt_metrics['Acc'][prompt_idx]),
                    Fscore=self.format_metric(
                        prompt_metrics['Fscore'][prompt_idx]),
                    Precision=self.format_metric(
                        prompt_metrics['Precision'][prompt_idx]),
                    Recall=self.format_metric(
                        prompt_metrics['Recall'][prompt_idx])))
        return metrics

    @staticmethod
    def format_metric(value):
        if np.isnan(value):
            return 'nan'
        return float(np.round(value * 100, 2))

    @staticmethod
    def intersect_and_union(pred_label: torch.Tensor,
                            label: torch.Tensor,
                            label_sets,
                            ignore_index: int,
                            class_scores: Optional[torch.Tensor] = None,
                            prompt_scores: Optional[torch.Tensor] = None,
                            prompt_metadata=None,
                            score_threshold: float = 0.1):
        valid_mask = label != ignore_index
        pred_label = pred_label[valid_mask]
        label = label[valid_mask]
        if class_scores is not None:
            class_scores = class_scores.to(label.device)
        if prompt_scores is not None:
            prompt_scores = prompt_scores.to(label.device)

        area_intersect = []
        area_union = []
        area_pred_label = []
        area_label = []
        for concept_idx, valid_labels in enumerate(label_sets):
            if class_scores is not None:
                pred_mask = class_scores[concept_idx][valid_mask] > score_threshold
            else:
                pred_mask = pred_label == concept_idx
            label_mask = torch.zeros_like(label, dtype=torch.bool)
            for valid_label in valid_labels:
                label_mask |= label == valid_label

            intersect = pred_mask & label_mask
            union = pred_mask | label_mask
            area_intersect.append(intersect.sum())
            area_union.append(union.sum())
            area_pred_label.append(pred_mask.sum())
            area_label.append(label_mask.sum())

        prompt_area_intersect = []
        prompt_area_union = []
        prompt_area_pred_label = []
        prompt_area_label = []
        for prompt_idx, metadata in enumerate(prompt_metadata or []):
            valid_labels = label_sets[metadata['concept_idx']]
            if prompt_scores is not None:
                prompt_mask = (
                    prompt_scores[prompt_idx][valid_mask] > score_threshold)
            else:
                prompt_mask = pred_label == metadata['concept_idx']
            label_mask = torch.zeros_like(label, dtype=torch.bool)
            for valid_label in valid_labels:
                label_mask |= label == valid_label

            intersect = prompt_mask & label_mask
            union = prompt_mask | label_mask
            prompt_area_intersect.append(intersect.sum())
            prompt_area_union.append(union.sum())
            prompt_area_pred_label.append(prompt_mask.sum())
            prompt_area_label.append(label_mask.sum())

        return (torch.stack(area_intersect).cpu(),
                torch.stack(area_union).cpu(),
                torch.stack(area_pred_label).cpu(),
                torch.stack(area_label).cpu(),
                torch.stack(prompt_area_intersect).cpu(),
                torch.stack(prompt_area_union).cpu(),
                torch.stack(prompt_area_pred_label).cpu(),
                torch.stack(prompt_area_label).cpu())

    @staticmethod
    def total_area_to_metrics(total_area_intersect, total_area_union,
                              total_area_pred_label, total_area_label,
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        total_area_intersect = total_area_intersect.numpy()
        total_area_union = total_area_union.numpy()
        total_area_pred_label = total_area_pred_label.numpy()
        total_area_label = total_area_label.numpy()

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            precision = total_area_intersect / total_area_pred_label
            recall = acc
            fscore = ((1 + beta**2) * precision * recall /
                      (beta**2 * precision + recall))
            a_acc = total_area_intersect.sum() / total_area_label.sum()

        ret_metrics = OrderedDict(
            aAcc=a_acc,
            IoU=iou,
            Acc=acc,
            Fscore=fscore,
            Precision=precision,
            Recall=recall)
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                key: np.nan_to_num(value, nan=nan_to_num)
                for key, value in ret_metrics.items()
            })
        return ret_metrics
