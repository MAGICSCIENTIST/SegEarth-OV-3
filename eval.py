import os
import os.path as osp
import argparse
import numpy as np
from mmengine.runner import Runner
from mmengine.config import Config, DictAction

import segearthov3_segmentor
import segearthov3_change_detector
import custom_datasets
import custom_transforms
import custom_visualizers
import custom_metrics
from custom_savers import save_tnls_outputs
from custom_visualizers import save_tnls_concept_comparisons
from metrics_reporting import MetricsReporter


def parse_args():
    parser = argparse.ArgumentParser(
        description='CorrCLIP evaluation with MMSeg')
    parser.add_argument('config', default='./configs/cfg_loveda.py')
    parser.add_argument(
        '--show',
        action='store_true',
        help='save visualization results to show_dir')
    parser.add_argument(
        '--show_dir',
        default=None,
        help='directory to save visualization images')
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display interval for visualization when --show is enabled')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def append_experiment_result(file_path,
                             experiment_data,
                             class_metrics=None,
                             prompt_metrics=None):
    first = experiment_data[0]
    MetricsReporter.write_workbook(
        file_path,
        summaries=experiment_data,
        class_metrics=class_metrics,
        prompt_metrics=prompt_metrics,
        model=first.get('Model'),
        dataset=first.get('Dataset'))


def to_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def format_metric(value):
    if np.isnan(value):
        return 'nan'
    return float(np.round(value * 100, 2))


def compute_class_metrics(metric, results):
    if not results:
        return []

    areas = tuple(zip(*results))
    area_intersect = sum(to_numpy(area) for area in areas[0])
    area_union = sum(to_numpy(area) for area in areas[1])
    area_pred_label = sum(to_numpy(area) for area in areas[2])
    area_label = sum(to_numpy(area) for area in areas[3])

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = area_intersect / area_union
        acc = area_intersect / area_label
        precision = area_intersect / area_pred_label
        recall = acc
        beta = getattr(metric, 'beta', 1)
        fscore = ((1 + beta**2) * precision * recall /
                  (beta**2 * precision + recall))

    classes = getattr(metric, 'dataset_meta', {}).get('classes', [])
    return [
        dict(
            Class=class_name,
            IoU=format_metric(iou[index]),
            Acc=format_metric(acc[index]),
            Fscore=format_metric(fscore[index]),
            Precision=format_metric(precision[index]),
            Recall=format_metric(recall[index]))
        for index, class_name in enumerate(classes)
    ]


def capture_class_metrics(runner):
    metrics = getattr(runner.test_evaluator, 'metrics', [])
    for metric in metrics:
        if (not hasattr(metric, 'compute_metrics')
                or not getattr(metric, 'supports_class_capture', True)):
            continue
        original_compute_metrics = metric.compute_metrics

        def wrapped_compute_metrics(results, metric=metric,
                                    original_compute_metrics=original_compute_metrics):
            metric._last_class_metrics = compute_class_metrics(metric, results)
            return original_compute_metrics(results)

        metric.compute_metrics = wrapped_compute_metrics


def attach_output_handlers(runner, result_dir, show_dir, score_threshold=0.1):
    metrics = getattr(runner.test_evaluator, 'metrics', [])
    concept_metrics = [
        metric for metric in metrics
        if hasattr(metric, 'concepts')
        and getattr(metric, 'supports_output_handlers', True)
    ]
    if not concept_metrics:
        return

    original_process = runner.test_evaluator.process

    def wrapped_process(data_samples, data_batch=None):
        output_samples = [
            data_sample.to_dict() if hasattr(data_sample, 'to_dict')
            else data_sample for data_sample in data_samples
        ]
        for metric in concept_metrics:
            for data_sample in output_samples:
                if getattr(metric, 'keep_results', True):
                    save_tnls_outputs(data_sample, metric.concepts,
                                      result_dir)
                save_tnls_concept_comparisons(
                    data_sample,
                    metric.concepts,
                    show_dir,
                    ignore_index=getattr(metric, 'ignore_index', 255),
                    score_threshold=score_threshold,
                    gt_alpha=getattr(metric, 'gt_alpha', 0.55))
        return original_process(data_samples, data_batch)

    runner.test_evaluator.process = wrapped_process


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # In this project, --show means saving visualizations, not opening a UI.
        visualization_hook['draw'] = args.show or args.show_dir is not None
        visualization_hook['show'] = False
        if args.show:
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()
    print(os.getcwd())
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    default_work_dir = osp.join(
        './work_dirs', osp.splitext(osp.basename(args.config))[0])
    output_root = args.out if args.out is not None else default_work_dir
    cfg.work_dir = output_root

    result_dir = osp.join(output_root, 'result')

    # Keep all eval artifacts under output_root; predictions go to result/.
    evaluator_cfgs = (cfg.test_evaluator if isinstance(cfg.test_evaluator, list)
                      else [cfg.test_evaluator])
    for evaluator_cfg in evaluator_cfgs:
        evaluator_cfg['output_dir'] = result_dir
        evaluator_cfg['keep_results'] = True
        evaluator_cfg.setdefault(
            'score_threshold', cfg.model.get('prob_thd', 0.1))
    if args.show_dir is None:
        args.show_dir = osp.join(output_root, 'vis')
    for evaluator_cfg in evaluator_cfgs:
        evaluator_cfg['vis_dir'] = args.show_dir

    os.makedirs(cfg.work_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    if args.show_dir:
        os.makedirs(args.show_dir, exist_ok=True)
    trigger_visualization_hook(cfg, args)
    runner = Runner.from_cfg(cfg)
    attach_output_handlers(
        runner,
        result_dir,
        args.show_dir,
        score_threshold=cfg.model.get('prob_thd', 0.1))
    capture_class_metrics(runner)
    results = runner.test()
    class_metrics = []
    prompt_metrics = []
    for metric in getattr(runner.test_evaluator, 'metrics', []):
        class_metrics.extend(getattr(metric, '_last_class_metrics', []))
        prompt_metrics.extend(getattr(metric, '_last_prompt_metrics', []))

    results.update({'Model': cfg.model.model_type,
                    'Dataset': cfg.dataset_type})

    if runner.rank == 0:
        append_experiment_result(osp.join(cfg.work_dir, 'results.xlsx'),
                                 [results], class_metrics, prompt_metrics)

    if runner.rank == 0:
        with open(osp.join(cfg.work_dir, 'results.txt'), 'a') as f:
            f.write(os.path.basename(args.config).split('.')[0] + '\n')
            for k, v in results.items():
                f.write(k + ': ' + str(v) + '\n')


if __name__ == '__main__':
    main()
