"""Shared metric conversion and Excel reporting utilities.

The helpers in this module are intentionally independent of MMSeg so they can
be reused by evaluation and the lightweight parameter-training entry points.
"""

from __future__ import annotations

import math
import os
import os.path as osp
from typing import Iterable, Mapping, Sequence

import openpyxl


SUMMARY_HEADERS = [
    'Model', 'Dataset', 'aAcc', 'mIoU', 'mAcc', 'mFscore',
    'mPrecision', 'mRecall', 'Epoch', 'Split', 'Loss',
]
CLASS_HEADERS = [
    'Model', 'Dataset', 'Class', 'IoU', 'Acc', 'Fscore', 'Precision',
    'Recall', 'TP', 'FP', 'FN', 'PredPixels', 'GTPixels', 'PredGTAreaRatio',
    'Epoch', 'Split', 'Threshold',
]
PROMPT_HEADERS = [
    'Model', 'Dataset', 'Prompt', 'ConceptID', 'ConceptName', 'IoU', 'Acc',
    'Fscore', 'Precision', 'Recall', 'TP', 'FP', 'FN', 'PredPixels',
    'GTPixels', 'PredGTAreaRatio', 'Epoch', 'Split', 'Threshold',
]


def safe_ratio(numerator, denominator):
    denominator = float(denominator)
    return float(numerator) / denominator if denominator else float('nan')


def counts_to_metrics(intersect, union, predicted, target, percent=True):
    """Convert aggregate binary pixel counts into a serializable metric row."""
    tp = float(intersect)
    pred = float(predicted)
    gt = float(target)
    fp = pred - tp
    fn = gt - tp
    scale = 100.0 if percent else 1.0
    precision = safe_ratio(tp, pred)
    recall = safe_ratio(tp, gt)
    fscore = safe_ratio(2.0 * precision * recall, precision + recall)
    return dict(
        IoU=safe_ratio(tp, union) * scale,
        Acc=recall * scale,
        Fscore=fscore * scale,
        Precision=precision * scale,
        Recall=recall * scale,
        TP=int(round(tp)),
        FP=int(round(fp)),
        FN=int(round(fn)),
        PredPixels=int(round(pred)),
        GTPixels=int(round(gt)),
        PredGTAreaRatio=safe_ratio(pred, gt),
    )


def _clean_value(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _ensure_sheet(workbook, name, headers):
    if name == 'summary':
        sheet = workbook[name] if name in workbook.sheetnames else workbook.active
        sheet.title = name
    else:
        sheet = (workbook[name] if name in workbook.sheetnames
                 else workbook.create_sheet(name))
    # Preserve existing files and columns; append any newly introduced fields.
    existing = [cell.value for cell in sheet[1]] if sheet.max_row else []
    if not any(existing):
        existing = []
    for header in headers:
        if header not in existing:
            existing.append(header)
    for column, header in enumerate(existing, 1):
        sheet.cell(1, column, header)
    return sheet, existing


def _append_rows(sheet, headers, rows, common):
    for row in rows or []:
        values = dict(common)
        values.update(row)
        sheet.append([_clean_value(values.get(header)) for header in headers])


def write_metrics_workbook(file_path: str,
                           summaries: Sequence[Mapping] | None = None,
                           class_metrics: Sequence[Mapping] | None = None,
                           prompt_metrics: Sequence[Mapping] | None = None,
                           model: str | None = None,
                           dataset: str | None = None):
    """Append metric rows while retaining legacy sheets and columns."""
    os.makedirs(osp.dirname(osp.abspath(file_path)), exist_ok=True)
    try:
        workbook = openpyxl.load_workbook(file_path)
    except FileNotFoundError:
        workbook = openpyxl.Workbook()
    common = {'Model': model, 'Dataset': dataset}
    if summaries:
        sheet, headers = _ensure_sheet(workbook, 'summary', SUMMARY_HEADERS)
        _append_rows(sheet, headers, summaries, common)
    if class_metrics:
        sheet, headers = _ensure_sheet(workbook, 'per_class', CLASS_HEADERS)
        _append_rows(sheet, headers, class_metrics, common)
    if prompt_metrics:
        sheet, headers = _ensure_sheet(workbook, 'per_prompt', PROMPT_HEADERS)
        _append_rows(sheet, headers, prompt_metrics, common)
    workbook.save(file_path)


class MetricsReporter:
    """Shared facade used by evaluation and parameter-training scripts."""

    counts_to_metrics = staticmethod(counts_to_metrics)
    write_workbook = staticmethod(write_metrics_workbook)
