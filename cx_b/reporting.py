"""CX-B console output and the two fixed text reports.

No trainer/model/entry-point imports. Metrics passed to this module are fractions;
human-readable ratios are percentages, counts are integers.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np


RATIOS = ('iou', 'precision', 'recall', 'f1')
COUNTS = ('TP', 'FP', 'FN', 'PredPixels', 'GTPixels')
FIELDS = (*RATIOS, *COUNTS)
TIMES = ('elapsed', 'eta', 'expected_finish')


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def binary_metrics(tp, fp, fn):
    def ratio(a, b):
        return float(a/b) if b else None
    return dict(IoU=ratio(tp, tp+fp+fn), Precision=ratio(tp, tp+fp),
                Recall=ratio(tp, tp+fn), Acc=ratio(tp, tp+fn),
                Fscore=ratio(2*tp, 2*tp+fp+fn), TP=int(tp), FP=int(fp), FN=int(fn),
                PredPixels=int(tp+fp), GTPixels=int(tp+fn))


def count_binary(probability, target, valid, threshold):
    pred = probability.detach()[valid] >= threshold
    truth = target[valid].bool()
    return np.array([int((pred & truth).sum()), int((pred & ~truth).sum()),
                     int((~pred & truth).sum())], dtype=np.int64)


def prediction_changes(before, after, target, valid):
    """Counts on the same valid pixels; accepts NumPy or Torch label arrays."""
    old_correct, new_correct = before == target, after == target
    return dict(changed_pixels=int(((before != after) & valid).sum()),
                corrected_pixels=int((~old_correct & new_correct & valid).sum()),
                damaged_pixels=int((old_correct & ~new_correct & valid).sum()))


def normalize(metrics, binary=False):
    if metrics is None:
        return dict.fromkeys(FIELDS)
    if binary:
        metrics = binary_metrics(metrics['TP'], metrics['FP'], metrics['FN'])
    result = {}
    for name, original in [('iou', 'IoU'), ('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'Fscore')]:
        value = metrics.get(original)
        result[name] = 100*float(value) if value is not None and math.isfinite(float(value)) else None
    result.update({key: metrics.get(key) for key in COUNTS})
    return result


def aggregate(rows):
    result = {}
    for key in FIELDS:
        values = [row[key] for row in rows if row.get(key) is not None]
        result[key] = (sum(values)/len(values) if key in RATIOS else sum(values)) if values else None
    return result


def summary_rows(prompts, concepts):
    """Build this evaluation's three summaries from its displayed detail rows.

    Ratios are macro means; counts are sums. Missing levels remain null, and
    prompts are averaged individually rather than first averaged by concept.
    """
    return [
        dict(level='multiclass', **aggregate([
            row for row in concepts if row['level'] == 'multiclass'])),
        dict(level='concept_mean', **aggregate([
            row for row in concepts if row['level'] == 'concept'])),
        dict(level='prompt_mean', **aggregate(prompts)),
    ]


class LayerCounts:
    """Count all three layers in the same pass, with no retained score maps."""
    def __init__(self, records, count, binary_threshold=.5, final_threshold=.5, background=0):
        self.records, self.count = records, count
        self.binary_threshold, self.final_threshold, self.background = binary_threshold, final_threshold, background
        self.prompts = np.zeros((len(records), 3), np.int64)
        self.concepts = np.zeros((count, 3), np.int64)
        self.confusion = np.zeros((count, count+1), np.int64)
        self.refined_concepts = None
        self.refiner_changes = None

    def update(self, scores, prompts, target, valid, raw_scores=None):
        import torch
        with torch.no_grad():
            values = scores.detach()
            if values.shape != (self.count, *target.shape) or not torch.isfinite(values).all():
                raise ValueError('Invalid score shape or nonfinite concept probabilities')
            if (values < 0).any() or (values > 1).any():
                raise ValueError('Concept probabilities must be in [0,1]')
            maximum, prediction = values.max(0)
            background = self.background if 0 <= self.background < self.count else self.count
            prediction = torch.where(maximum < self.final_threshold, background, prediction)
            indices = target[valid]*(self.count+1)+prediction[valid]
            self.confusion += torch.bincount(indices, minlength=self.count*(self.count+1)).reshape(
                self.count, self.count+1).cpu().numpy()
            for c in range(self.count):
                self.concepts[c] += count_binary(values[c] if raw_scores is None else raw_scores[c],
                                                target == c, valid, self.binary_threshold)
            if raw_scores is not None:
                if self.refined_concepts is None:
                    self.refined_concepts = np.zeros_like(self.concepts)
                    self.refiner_changes = dict(changed_pixels=0, corrected_pixels=0, damaged_pixels=0)
                raw_maximum, raw_prediction = raw_scores.max(0)
                raw_prediction = torch.where(raw_maximum < self.final_threshold, background, raw_prediction)
                for key, value in prediction_changes(raw_prediction, prediction, target, valid).items():
                    self.refiner_changes[key] += value
                for c in range(self.count):
                    self.refined_concepts[c] += count_binary(values[c], target == c, valid, self.binary_threshold)
            for index, record in enumerate(self.records):
                self.prompts[index] += count_binary(prompts[index], target == record['concept_index'], valid,
                                                   self.binary_threshold)

    def result(self):
        matrix = self.confusion
        tp, gt, pred = np.diag(matrix), matrix.sum(1), matrix[:, :self.count].sum(0)
        rows = []
        for c in range(self.count):
            row = binary_metrics(tp[c], pred[c]-tp[c], gt[c]-tp[c])
            # Match MMSeg final multiclass F-score (undefined P/R or 0/0 => null).
            precision, recall = row['Precision'], row['Recall']
            row['Fscore'] = (None if precision is None or recall is None or precision+recall == 0
                             else 2*precision*recall/(precision+recall))
            rows.append(dict(row, concept_index=c))
        def mean(key):
            values = [r[key] for r in rows if r[key] is not None]
            return float(np.mean(values)) if values else None
        result = dict(mIoU=mean('IoU'), mAcc=mean('Acc'), mPrecision=mean('Precision'),
            mRecall=mean('Recall'), mFscore=mean('Fscore'),
            aAcc=float(tp.sum()/matrix.sum()) if matrix.sum() else None,
            valid_pixels=int(matrix.sum()), unassigned_pixels=int(matrix[:, -1].sum()),
            per_class=rows, confusion_matrix=matrix.tolist(),
            prompt_binary=[binary_metrics(*row) for row in self.prompts],
            concept_binary=[binary_metrics(*row) for row in self.concepts])
        if self.refined_concepts is not None:
            result['refined_concept_binary'] = [binary_metrics(*row) for row in self.refined_concepts]
            result['refiner_changes'] = dict(self.refiner_changes)
        return result


class _TimedStream:
    """Timestamp ordinary prints from dependencies without creating legacy logs."""
    def __init__(self, reporter, stream):
        self.reporter, self.stream, self.buffer = reporter, stream, ''

    def write(self, text):
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip():
                line = re.sub(r'\b(?:elapsed|eta|expected_finish)=[^\s]+(?: [0-9]{2}:[0-9]{2}:[0-9]{2})?', '', line).rstrip()
                self.stream.write(line+' '+self.reporter.time_text()+'\n')
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class CXBReporter:
    def __init__(self, output_dir, config, args=None):
        self.root, self.config, self.args = Path(output_dir), config, args or {}
        self.started = time.monotonic()
        self.console = None
        self.stderr = None
        self.active = False
        self.started_output = False
        self.files_started = False
        self.parameter_text = ''
        self.records, self.concepts = [], []
        self.plan, self.progress = {}, {}
        self.context = None
        self.done = False
        self.final_result = None
        self.threshold = float(config['cx_b'].get('metric_threshold', .5))

    def time_values(self):
        elapsed = time.monotonic()-self.started
        total, completed = sum(self.plan.values()), sum(self.progress.values())
        remaining = (0. if self.done else elapsed*(total-completed)/completed
                     if completed > 0 and total >= completed else None)
        return dict(elapsed=duration(elapsed), eta=duration(remaining) if remaining is not None else 'pending',
                    expected_finish=(datetime.now(timezone(timedelta(hours=8)))+timedelta(seconds=remaining)).strftime('%Y-%m-%d %H:%M:%S')
                    if remaining is not None else 'pending')

    def time_text(self):
        values = self.time_values()
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
        return (' '.join(f'{key}={value}' for key, value in values.items())
                + f' timestamp={stamp} run_eta={values["eta"]} run_expected_finish={values["expected_finish"]}')

    def raw(self, text):
        stream = self.console or sys.stdout
        stream.write(text)
        stream.flush()

    def line(self, message):
        self.raw(message+' '+self.time_text()+'\n')

    def start(self, write_files=True):
        if self.started_output:
            if write_files:
                self.begin_run()
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.started_output = True
        cx, model = self.config['cx_b'], self.config.get('model', {})
        concept_path = self.config.get('concept_path', model.get('classname_path', ''))
        header = (f'[run] mode={cx.get("joint_mode", "independent")} binary_threshold={self.threshold} '
                  f'final_threshold={model.get("prob_thd", .1)} background={cx.get("prediction_background", 65535)} '
                  f'ignore_index={255 if str(concept_path).lower().endswith(".json") else 0} metric_unit=percent '
                  'timezone=Asia/Shanghai eta_scope=run')
        blocks = [header+' '+self.time_text()+'\n']
        definitions = dict(unit='percent', display_decimals=4,
            prompt='single prompt binary', concept='within-concept aggregation binary',
            multiclass='final class competition and background fallback',
            summary_ratios='macro_mean', summary_counts='sum',
            null='not evaluated, not applicable, or undefined; not an update/freeze flag',
            eta='whole-run estimate from completed planned epoch work; pending until measurable')
        for name, value in [('args', self.args), ('config', self.config), ('metric_definition', definitions)]:
            blocks += [f'\n[{name}] {self.time_text()}\n', json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n']
        self.parameter_text = ''.join(blocks)
        self.raw(self.parameter_text)
        if write_files:
            self.begin_run()

    def begin_run(self):
        """Persist one actual run; skip-complete checks must not rewrite reports."""
        if self.files_started:
            return
        if not self.started_output:
            self.start(write_files=False)
        (self.root/'log_fix.txt').write_text(self.parameter_text, encoding='utf-8')
        (self.root/'val_fix.txt').write_text('', encoding='utf-8')
        self.files_started = True

    def __enter__(self):
        self.console, self.stderr = sys.stdout, sys.stderr
        self.start(write_files=False)
        self.active = True
        sys.stdout = _TimedStream(self, self.console)
        sys.stderr = _TimedStream(self, self.stderr)
        return self

    def __exit__(self, kind, value, traceback):
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, _TimedStream) and stream.buffer.strip():
                stream.write('\n')
        sys.stdout, sys.stderr = self.console, self.stderr
        self.active = False
        return False

    def catalog(self, concepts, records):
        self.concepts, self.records = concepts, records

    def set_plan(self, plan):
        self.plan = dict(plan)
        self.progress = {key: 0. for key in plan}

    def advance(self, key, work):
        if key in self.plan:
            self.progress[key] = min(float(work), self.plan[key])

    def finish_phase(self, key):
        self.advance(key, self.plan.get(key, 0))

    def phase(self, stage, module=None):
        self.start()
        key = (stage, module)
        if key != self.context:
            self.context = key
            message = f'[stage={stage}'+(f' module={module}' if module is not None else '')+']'
            self.line(message)
            with (self.root/'val_fix.txt').open('a', encoding='utf-8') as stream:
                stream.write(message+' '+self.time_text()+'\n')

    def tables(self, prompt_metrics=None, concept_metrics=None, multiclass_metrics=None):
        prompt_metrics, concept_metrics, multiclass_metrics = prompt_metrics or {}, concept_metrics or {}, multiclass_metrics or {}
        prompts = [dict(prompt=r['prompt'], concept=r['concept_id'], **normalize(prompt_metrics.get(r['prompt_index']), True))
                   for r in self.records]
        concepts = []
        for i, concept in enumerate(self.concepts):
            for level, mapping in [('concept', concept_metrics), ('multiclass', multiclass_metrics)]:
                concepts.append(dict(concept=concept['concept_id'], level=level,
                                     **normalize(mapping.get(i), level == 'concept')))
        return prompts, concepts, summary_rows(prompts, concepts)

    @staticmethod
    def mappings(metrics):
        return (dict(enumerate(metrics.get('prompt_binary', []))),
                dict(enumerate(metrics.get('concept_binary', []))),
                dict(enumerate(metrics.get('per_class', []))))

    def progress_line(self, split, epoch, epochs, tables, loss=None, step=None, samples=None):
        summary = tables[2]
        def number(value):
            return 'null' if value is None else f'{value:.4f}'
        label = f'[{split} epoch={epoch}/{epochs}'
        if step is not None:
            label += f' step={step}/{samples if samples is not None else "?"}'
        self.line(label+f'] loss={number(loss)} iou_prompt={number(summary[2]["iou"])} '
                  f'iou_concept={number(summary[1]["iou"])} iou={number(summary[0]["iou"])}')

    def validation(self, epoch, epochs, tables, final=False, best_epoch=None):
        self.start()
        # Always emit the summary of these exact details, including final best
        # evaluation. Never rely on an optional or stale caller-supplied summary.
        prompts, concepts = tables[:2]
        summary = summary_rows(prompts, concepts)
        times = self.time_values()
        timing = self.time_text()
        marker = f'[final best_epoch={best_epoch}]' if final else f'[epoch={epoch}/{epochs}]'
        chunks = [marker+' '+timing+'\n']
        for name, identifiers, rows in [('prompt', ('prompt', 'concept'), prompts),
                                       ('concept', ('concept', 'level'), concepts),
                                       ('summary', ('level',), summary)]:
            chunks.append(f'\n[{name}] {timing}\n')
            csv_buffer = io.StringIO(newline='')
            fields = (*identifiers, *FIELDS, *TIMES)
            writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for row in rows:
                formatted = {key: 'null' if value is None else f'{value:.4f}' if key in RATIOS else value
                             for key, value in row.items()}
                writer.writerow(dict(formatted, **times))
            chunks.append(csv_buffer.getvalue())
        text = ''.join(chunks)+'\n'
        self.raw(text)
        with (self.root/'val_fix.txt').open('a', encoding='utf-8') as stream:
            stream.write(text)

    def binary(self, stage, index, epochs, samples=None):
        self.start()
        name = self.records[index]['prompt'] if stage == 'prompt' else self.concepts[index]['concept_id']
        self.phase(stage, name)
        return BinaryObserver(self, stage, index, epochs, samples)


class BinaryObserver:
    def __init__(self, reporter, stage, index, epochs, samples):
        self.reporter, self.stage, self.index, self.epochs, self.samples = reporter, stage, index, epochs, samples
        self.key = f'{stage}_{index}'
        self.prompt_counts = {'train': {}, 'val': {}}

    def reset_prompts(self, split):
        self.prompt_counts[split] = {}

    def prompt_update(self, split, index, probability, target, valid):
        counts = self.prompt_counts[split].setdefault(index, np.zeros(3, np.int64))
        counts += count_binary(probability, target, valid, self.reporter.threshold)

    def tables(self, metrics, split):
        prompts = {i: binary_metrics(*v) for i, v in self.prompt_counts[split].items()}
        concepts = {}
        (prompts if self.stage == 'prompt' else concepts)[self.index] = metrics
        return self.reporter.tables(prompts, concepts)

    def validation(self, epoch, metrics):
        self.reporter.advance(self.key, epoch+1)
        tables = self.tables(metrics, 'val')
        self.reporter.progress_line('val', epoch, self.epochs, tables)
        self.reporter.validation(epoch, self.epochs, tables)

    def train(self, epoch, step, metrics, loss):
        self.reporter.advance(self.key, epoch+(step/self.samples if self.samples else 0))
        self.reporter.progress_line('train', epoch, self.epochs, self.tables(metrics, 'train'), loss, step, self.samples)

    def selected(self, epoch, metrics):
        self.reporter.finish_phase(self.key)
        summary = self.reporter.tables({self.index: metrics} if self.stage=='prompt' else {},
                                       {self.index: metrics} if self.stage=='concept' else {})[2]
        values = [summary[2]['iou'], summary[1]['iou'], summary[0]['iou']]
        self.reporter.line(f'[selected best_epoch={epoch}] '+ ' '.join(
            f'{key}={"null" if value is None else f"{value:.4f}"}'
            for key, value in zip(('iou_prompt', 'iou_concept', 'iou'), values)))


class JointObserver:
    def __init__(self, reporter, epochs, samples):
        self.reporter, self.epochs, self.samples = reporter, epochs, samples
        reporter.phase('joint')

    def validation(self, epoch, metrics):
        self.reporter.advance('joint', epoch+1)
        tables = self.reporter.tables(*self.reporter.mappings(metrics))
        self.reporter.progress_line('val', epoch, self.epochs, tables)
        self.reporter.validation(epoch, self.epochs, tables)
        if 'base_metrics' in metrics:
            self.reporter.line(f'[refiner validation] base_mIoU={100*metrics["base_metrics"]["mIoU"]:.6f} '
                               f'refined_mIoU={100*metrics["mIoU"]:.6f} '
                               f'gain_pp={100*metrics["refiner_mIoU_gain"]:.6f} '
                               + ' '.join(f'{k}={v}' for k,v in metrics['refiner_changes'].items()))

    def train(self, epoch, step, metrics, loss):
        self.reporter.advance('joint', epoch+(step/self.samples if self.samples else 0))
        self.reporter.progress_line('train', epoch, self.epochs,
            self.reporter.tables(*self.reporter.mappings(metrics)), loss, step, self.samples)
