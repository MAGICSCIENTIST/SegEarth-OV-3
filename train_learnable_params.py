import argparse
import hashlib
import json
import os
import os.path as osp
import runpy
import shutil
import subprocess
import time
from types import SimpleNamespace

os.environ.setdefault('SEGOV3_LIGHTWEIGHT_TRAIN', '1')
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from segearthov3_learnable_segmentor import SegEarthOV3LearnableSegmentation
from metrics_reporting import MetricsReporter


def load_chinese_font(size=18):
    font_candidates = [
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\simsun.ttc',
        r'C:\Windows\Fonts\arialuni.ttf',
    ]
    for font_path in font_candidates:
        if osp.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def label_color(label):
    label = int(label)
    return np.array([
        (37 * label + 17) % 256,
        (67 * label + 29) % 256,
        (97 * label + 43) % 256
    ],
                    dtype=np.uint8)


def concept_color(index):
    index = int(index) + 1
    return np.array([
        (53 * index + 31) % 256,
        (97 * index + 47) % 256,
        (149 * index + 61) % 256
    ],
                    dtype=np.uint8)


def concept_gt_color(concept, fallback_index):
    labels = concept.get('tnls_dl3', [])
    if labels:
        return label_color(labels[0])
    return concept_color(fallback_index)


def safe_filename(text, max_length=120):
    keep = []
    for char in str(text):
        if char in '<>:"/\\|?*' or ord(char) < 32:
            keep.append('_')
        elif char.isspace():
            keep.append('_')
        else:
            keep.append(char)
    name = ''.join(keep).strip(' ._')
    return (name[:max_length].rstrip(' ._') or 'item')


def font_supports_text(font, text):
    try:
        font.getbbox(text)
        return True
    except UnicodeEncodeError:
        return False


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__


class LightweightDataSample:
    def __init__(self, img_path, gt):
        self.metainfo = {'img_path': img_path}
        self.gt_sem_seg = SimpleNamespace(data=gt)


class LightweightSegDataset:
    IMG_SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')

    def __init__(self, dataset_cfg):
        self.data_root = dataset_cfg.get('data_root', '')
        self.data_prefix = dataset_cfg['data_prefix']
        self.img_dir = self._full_path(self.data_prefix['img_path'])
        self.ann_dir = self._full_path(self.data_prefix['seg_map_path'])
        self.img_suffix = dataset_cfg.get('img_suffix', None)
        self.seg_map_suffix = dataset_cfg.get('seg_map_suffix', None)
        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(f'No training images found in {self.img_dir}')

    def _full_path(self, path):
        if osp.isabs(path):
            return path
        return osp.join(self.data_root, path)

    def _load_samples(self):
        samples = []
        for root, _, files in os.walk(self.img_dir):
            for filename in files:
                ext = osp.splitext(filename)[1].lower()
                if self.img_suffix is not None:
                    if not filename.endswith(self.img_suffix):
                        continue
                elif ext not in self.IMG_SUFFIXES:
                    continue
                img_path = osp.join(root, filename)
                rel_path = osp.relpath(img_path, self.img_dir)
                ann_path = self._find_annotation(rel_path)
                if ann_path is not None:
                    samples.append((img_path, ann_path))
        return sorted(samples)

    def _find_annotation(self, rel_img_path):
        stem, img_ext = osp.splitext(rel_img_path)
        candidates = []
        if self.seg_map_suffix is not None:
            candidates.append(stem + self.seg_map_suffix)
        candidates.append(rel_img_path)
        candidates.extend(stem + ext for ext in self.IMG_SUFFIXES)
        for candidate in candidates:
            ann_path = osp.join(self.ann_dir, candidate)
            if osp.exists(ann_path):
                return ann_path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, ann_path = self.samples[index]
        gt = torch.from_numpy(np.array(Image.open(ann_path))).long()
        if gt.dim() == 3:
            gt = gt[..., 0]
        return {'data_samples': LightweightDataSample(img_path, gt)}


def render_gt_label(label, ignore_index):
    label_np = label.detach().cpu().numpy()
    image = np.zeros((*label_np.shape, 3), dtype=np.uint8)
    for value in np.unique(label_np):
        if int(value) == ignore_index:
            continue
        image[label_np == value] = label_color(value)
    return image


def load_raster(image_path, shape):
    raster = Image.open(image_path).convert('RGB')
    raster = raster.resize((shape[1], shape[0]))
    return np.asarray(raster, dtype=np.uint8)


def blend_overlay(raster, color_layer, mask, alpha):
    overlay = raster.copy()
    blended = (
        raster.astype(np.float32) * (1 - alpha) +
        color_layer.astype(np.float32) * alpha)
    overlay[mask] = blended[mask].astype(np.uint8)
    return overlay


def render_gt_overlay(raster, label, ignore_index, alpha):
    label_np = label.detach().cpu().numpy()
    label_vis = render_gt_label(label, ignore_index)
    valid_mask = label_np != ignore_index
    return blend_overlay(raster, label_vis, valid_mask, alpha)


def render_prediction(raster, pred_label, concepts, alpha):
    pred_np = pred_label.detach().cpu().numpy()
    color_layer = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
    valid_mask = np.zeros(pred_np.shape, dtype=bool)
    for concept_idx, concept in enumerate(concepts):
        concept_mask = pred_np == concept_idx
        color_layer[concept_mask] = concept_gt_color(concept, concept_idx)
        valid_mask |= concept_mask
    return blend_overlay(raster, color_layer, valid_mask, alpha)


def render_concept_mask(raster, pred_label, concept_idx, concept, alpha):
    pred_np = pred_label.detach().cpu().numpy()
    mask = pred_np == concept_idx
    color_layer = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
    color_layer[mask] = concept_gt_color(concept, concept_idx)
    return blend_overlay(raster, color_layer, mask, alpha)


def save_compare_image(left, right, left_title, right_title, out_file, font):
    title_height = 32
    h, w = left.shape[:2]
    canvas = np.full((h + title_height, w * 2, 3), 255, dtype=np.uint8)
    canvas[title_height:, :w] = left
    canvas[title_height:, w:] = right
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    if not font_supports_text(font, right_title):
        right_title = ''.join(char if ord(char) < 128 else '_'
                              for char in right_title)
    draw.text((6, 5), left_title, fill=(0, 0, 0), font=font)
    draw.text((w + 6, 5), right_title, fill=(0, 0, 0), font=font)
    image.save(out_file)


def safe_divide(numerator, denominator):
    numerator = numerator.float()
    denominator = denominator.float()
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.full_like(numerator, float('nan'), dtype=torch.float32))


def format_metric_value(value):
    value = float(value)
    if np.isnan(value):
        return 'nan'
    return str(float(np.round(value * 100, 2)))


def make_table(headers, rows):
    try:
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = headers
        for row in rows:
            table.add_row(row)
        return table.get_string()
    except ImportError:
        widths = [
            max(len(str(item)) for item in [header] + [
                row[index] for row in rows
            ])
            for index, header in enumerate(headers)
        ]
        sep = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'
        lines = [sep]
        lines.append('|' + '|'.join(
            f' {str(header).center(widths[index])} '
            for index, header in enumerate(headers)) + '|')
        lines.append(sep)
        for row in rows:
            lines.append('|' + '|'.join(
                f' {str(value).center(widths[index])} '
                for index, value in enumerate(row)) + '|')
        lines.append(sep)
        return '\n'.join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train only SegEarthOV3 learnable calibration parameters.')
    parser.add_argument('config', help='Path to a SegEarthOV3 learnable config.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max-iters', type=int, default=None)
    parser.add_argument(
        '--max-val-samples',
        type=int,
        default=None,
        help='Maximum validation samples per validation epoch. -1 means all.')
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=None,
        help='Console progress print interval in processed samples.')
    parser.add_argument(
        '--param-reg-weight',
        type=float,
        default=None,
        help='L2 regularization weight that keeps calibration parameters near defaults.')
    parser.add_argument(
        '--grad-clip-norm',
        type=float,
        default=None,
        help='Clip calibration parameter gradients by global norm. 0 disables clipping.')
    parser.add_argument(
        '--concept-weight-min',
        type=float,
        default=None,
        help='Deprecated compatibility option; concept_weight is fixed.')
    parser.add_argument(
        '--concept-weight-max',
        type=float,
        default=None,
        help='Deprecated compatibility option; concept_weight is fixed.')
    parser.add_argument(
        '--prompt-weight-max',
        type=float,
        default=None,
        help='Maximum prompt_weight after each optimizer step.')
    parser.add_argument(
        '--presence-gain-max',
        type=float,
        default=None,
        help='Maximum per-prompt presence_gain after each optimizer step.')
    parser.add_argument(
        '--loss-mode',
        choices=['ce', 'mixed', 'concept_metric'],
        default=None,
        help=(
            'Loss preset name. ce keeps the original cross entropy loss; '
            'mixed uses CE plus concept metric losses; concept_metric is '
            'intended for metric-only experiments when CE weight is 0.'))
    parser.add_argument(
        '--ce-loss-weight',
        type=float,
        default=None,
        help='Weight for pixel-wise cross entropy over concept labels.')
    parser.add_argument(
        '--soft-iou-loss-weight',
        type=float,
        default=None,
        help='Weight for differentiable per-concept soft IoU loss.')
    parser.add_argument(
        '--recall-loss-weight',
        type=float,
        default=None,
        help='Weight for differentiable per-concept recall floor loss.')
    parser.add_argument(
        '--recall-target',
        type=float,
        default=None,
        help='Target recall used by the recall floor loss.')
    parser.add_argument(
        '--worst-concept-loss-weight',
        type=float,
        default=None,
        help=(
            'Extra weight for the worst per-concept metric loss in each '
            'sample. This makes weak concepts harder to ignore.'))
    parser.add_argument(
        '--soft-iou-present-only',
        action='store_true',
        default=None,
        help='Average soft IoU loss only over concepts present in the sample.')
    parser.add_argument(
        '--soft-loss-eps',
        type=float,
        default=None,
        help='Numerical epsilon for differentiable metric losses.')
    parser.add_argument('--save-dir', default='weights/my')
    parser.add_argument(
        '--save-every-epochs',
        type=int,
        default=None,
        help='Save calibration parameters every N epochs. 0 disables periodic saving.')
    parser.add_argument(
        '--validate-every-epochs',
        type=int,
        default=None,
        help='Run validation every N epochs. 0 disables validation.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action='append',
        help='Override config options, e.g. model.prompt_fusion_mode=weighted_sum')
    return parser.parse_args()


TRAIN_ARG_DEFAULTS = {
    'epochs': 1,
    'lr': 1e-2,
    'max_iters': -1,
    'max_val_samples': -1,
    'progress_interval': 10,
    'param_reg_weight': 1e-3,
    'grad_clip_norm': 1.0,
    'concept_weight_min': 0.2,
    'concept_weight_max': 3.0,
    'prompt_weight_max': 3.0,
    'presence_gain_max': 8.0,
    'loss_mode': 'ce',
    'ce_loss_weight': 1.0,
    'soft_iou_loss_weight': 0.0,
    'recall_loss_weight': 0.0,
    'recall_target': 0.3,
    'worst_concept_loss_weight': 0.0,
    'soft_iou_present_only': False,
    'soft_loss_eps': 1e-6,
    'save_every_epochs': 0,
    'validate_every_epochs': 1,
}


def apply_learnable_train_config(args, cfg):
    train_cfg = cfg.get('learnable_train_cfg', {})
    for key, default_value in TRAIN_ARG_DEFAULTS.items():
        cli_value = getattr(args, key)
        if cli_value is not None:
            continue
        setattr(args, key, train_cfg.get(key, default_value))

    allowed_loss_modes = {'ce', 'mixed', 'concept_metric'}
    if args.loss_mode not in allowed_loss_modes:
        raise ValueError(
            f'Invalid learnable_train_cfg.loss_mode={args.loss_mode!r}. '
            f'Expected one of {sorted(allowed_loss_modes)}.')


def build_summary_writer(log_dir):
    if log_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        print(
            'TensorBoard logging disabled: install tensorboard in this '
            f'environment to enable it. ({exc})')
        return None
    os.makedirs(log_dir, exist_ok=True)
    print(f'Writing TensorBoard logs to {log_dir}')
    return SummaryWriter(log_dir=log_dir)


def describe_loss_args(args):
    return (
        f"loss_mode={args.loss_mode} "
        f"ce={args.ce_loss_weight:g} "
        f"soft_iou={args.soft_iou_loss_weight:g} "
        f"recall={args.recall_loss_weight:g} "
        f"recall_target={args.recall_target:g} "
        f"worst_concept={args.worst_concept_loss_weight:g} "
        f"soft_iou_present_only={args.soft_iou_present_only}")


def format_duration(seconds):
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def cuda_status():
    if not torch.cuda.is_available():
        return 'gpu_mem=n/a'
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return (
        f'gpu_mem={allocated:.2f}G reserved={reserved:.2f}G '
        f'peak={peak:.2f}G')


def progress_line(stage, epoch, done, total, loss, elapsed, total_epochs=None):
    total = max(int(total), 1)
    done = max(int(done), 0)
    progress = min(done / total, 1.0)
    sec_per_sample = elapsed / max(done, 1)
    eta = sec_per_sample * max(total - done, 0)
    rate = done / max(elapsed, 1e-6)
    epoch_text = f'epoch={epoch}'
    if total_epochs is not None:
        epoch_text = f'epoch={epoch}/{total_epochs}'
    return (
        f'[{stage}] {epoch_text} {done}/{total} '
        f'({progress * 100:.1f}%) loss={loss:.6f} '
        f'elapsed={format_duration(elapsed)} eta={format_duration(eta)} '
        f'{sec_per_sample:.2f}s/sample {rate:.2f} sample/s {cuda_status()}')


def get_gt(data_sample, device, label_map=None, ignore_index=255):
    gt = data_sample.gt_sem_seg.data
    if gt.dim() == 3:
        gt = gt.squeeze(0)
    gt = gt.long()
    if label_map is not None:
        mapped_gt = torch.full_like(gt, int(ignore_index))
        valid = (gt >= 0) & (gt < label_map.numel())
        mapped_gt[valid] = label_map[gt[valid]]
        gt = mapped_gt
    return gt.to(device)


def resize_scores(class_scores, target_shape):
    if class_scores.shape[-2:] == target_shape:
        return class_scores
    return F.interpolate(
        class_scores.unsqueeze(0),
        size=target_shape,
        mode='bilinear',
        align_corners=False).squeeze(0)


def class_scores_to_prediction(model, class_scores):
    params = model._bounded_prompt_values()
    seg_pred = torch.argmax(class_scores, dim=0)
    max_vals = class_scores.max(0)[0]
    thresholds = params['concept_threshold'].view(-1, 1, 1)
    concept_thresholds = torch.gather(
        thresholds.expand(-1, *seg_pred.shape),
        0,
        seg_pred.unsqueeze(0)).squeeze(0)
    global_threshold = torch.tensor(
        model.prob_thd, device=class_scores.device, dtype=max_vals.dtype)
    seg_pred[max_vals < torch.maximum(
        concept_thresholds, global_threshold)] = model.bg_idx
    return seg_pred


def save_validation_visualization(model,
                                  sample,
                                  class_scores,
                                  save_dir,
                                  epoch,
                                  ignore_index=255,
                                  gt_alpha=0.55):
    if save_dir is None:
        return
    data_sample = sample['data_samples']
    image_path = data_sample.metainfo['img_path']
    gt_label = data_sample.gt_sem_seg.data.squeeze()
    pred_label = class_scores_to_prediction(model, class_scores).detach().cpu()

    vis_root = osp.join(save_dir, 'vis_data')
    vis_image_dir = osp.join(vis_root, 'vis_image')
    visual_dir = osp.join(vis_root, 'visual')
    os.makedirs(vis_image_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    epoch_suffix = f'epoch_{epoch:04d}'
    basename = osp.splitext(osp.basename(image_path))[0]
    font = load_chinese_font()
    image_shape = gt_label.shape[-2:]
    raster = load_raster(image_path, image_shape)

    gt_vis = render_gt_overlay(raster, gt_label, ignore_index, gt_alpha)
    pred_vis = render_prediction(raster, pred_label, model.concepts, gt_alpha)
    save_compare_image(
        gt_vis, pred_vis, 'GT label', 'prediction',
        osp.join(vis_image_dir, f'{basename}_{epoch_suffix}.jpg'), font)

    for concept_idx, concept in enumerate(model.concepts):
        concept_id = safe_filename(concept.get('concept_id') or concept_idx)
        concept_name = concept.get('concept_name') or concept.get(
            'concept_id') or str(concept_idx)
        concept_vis = render_concept_mask(
            raster, pred_label, concept_idx, concept, gt_alpha)
        save_compare_image(
            gt_vis, concept_vis, 'GT label', concept_name,
            osp.join(
                visual_dir,
                f'{basename}_{concept_id}_{epoch_suffix}.jpg'), font)


def to_attr_dict(value):
    if isinstance(value, dict):
        return AttrDict({key: to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_attr_dict(item) for item in value]
    return value


def load_config(path):
    cfg_vars = runpy.run_path(path)
    cfg = {
        key: value
        for key, value in cfg_vars.items()
        if not key.startswith('_')
    }
    return to_attr_dict(cfg)


def parse_override_value(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered == 'true':
            return True
        if lowered == 'false':
            return False
        return value


def merge_cfg_options(cfg, cfg_options):
    if not cfg_options:
        return
    for group in cfg_options:
        for option in group:
            if '=' not in option:
                raise ValueError(f'Invalid cfg option: {option}')
            key, value = option.split('=', 1)
            target = cfg
            parts = key.split('.')
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = parse_override_value(value)


def build_model(model_cfg):
    kwargs = dict(model_cfg)
    kwargs.pop('type', None)
    kwargs.pop('model_type', None)
    return SegEarthOV3LearnableSegmentation(**kwargs)


def resolve_config_path(path, config_path=None):
    if path is None:
        return None
    path = osp.expanduser(str(path))
    if osp.isabs(path):
        return osp.abspath(path)

    cwd_path = osp.abspath(path)
    if osp.exists(cwd_path):
        return cwd_path

    if config_path is not None:
        config_dir_path = osp.abspath(
            osp.join(osp.dirname(osp.abspath(config_path)), path))
        if osp.exists(config_dir_path):
            return config_dir_path
    return cwd_path


def json_safe(value):
    if isinstance(value, AttrDict):
        value = dict(value)
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
            if not callable(item)
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def get_concept_json_path(cfg, config_path):
    candidates = [
        cfg.get('concept_path'),
        cfg.get('model', {}).get('classname_path'),
        cfg.get('test_evaluator', {}).get('concept_path'),
    ]
    for candidate in candidates:
        resolved = resolve_config_path(candidate, config_path)
        if resolved is not None and osp.exists(resolved):
            return resolved
    return None


def copy_snapshot_file(src, dst):
    if src is None or not osp.exists(src):
        return False
    if osp.abspath(src) == osp.abspath(dst):
        return False
    shutil.copy2(src, dst)
    return True


def save_experiment_snapshot(args, cfg):
    snapshot_dir = osp.join(args.save_dir, 'config_snapshot')
    os.makedirs(snapshot_dir, exist_ok=True)

    config_path = resolve_config_path(args.config)
    copied_config = copy_snapshot_file(
        config_path, osp.join(snapshot_dir, 'config_used.py'))

    concept_json_path = get_concept_json_path(cfg, args.config)
    copied_concept = False
    if concept_json_path is not None:
        copied_concept = copy_snapshot_file(
            concept_json_path,
            osp.join(snapshot_dir, osp.basename(concept_json_path)))

    repo_root = osp.dirname(osp.abspath(__file__))

    def git_output(*git_args):
        try:
            return subprocess.check_output(
                ['git', '-c', f'safe.directory={repo_root}', *git_args],
                cwd=repo_root,
                stderr=subprocess.STDOUT,
                text=True).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            return f'unavailable: {error}'

    source_paths = [
        osp.abspath(__file__),
        osp.join(repo_root, 'train_prompt_params_independent.py'),
        osp.join(repo_root, 'segearthov3_learnable_segmentor.py'),
        config_path,
        concept_json_path,
    ]
    source_hashes = {}
    for path in source_paths:
        if path is None or not osp.isfile(path):
            continue
        with open(path, 'rb') as source_file:
            source_hashes[osp.relpath(path, repo_root)] = hashlib.sha256(
                source_file.read()).hexdigest()

    resolved = dict(
        config_path=config_path,
        config_copied=copied_config,
        concept_json_path=concept_json_path,
        concept_json_copied=copied_concept,
        train_args={key: json_safe(value)
                    for key, value in vars(args).items()},
        config=json_safe(cfg),
        git_commit=git_output('rev-parse', 'HEAD'),
        git_status=git_output('status', '--short'),
        source_sha256=source_hashes)
    resolved_path = osp.join(snapshot_dir, 'resolved_config.json')
    with open(resolved_path, 'w', encoding='utf-8') as f:
        json.dump(resolved, f, ensure_ascii=False, indent=2)
    with open(osp.join(args.save_dir, 'config_snapshot.yaml'), 'w',
              encoding='utf-8') as f:
        # JSON is valid YAML and avoids adding a runtime YAML dependency.
        json.dump(resolved, f, ensure_ascii=False, indent=2)
    print(f'Saved experiment config snapshot to {snapshot_dir}')


def get_train_dataset_cfg(cfg):
    if 'train_dataloader' in cfg:
        return cfg.train_dataloader.dataset
    if 'test_dataloader' in cfg:
        print('train_dataloader is not set; falling back to test_dataloader.')
        return cfg.test_dataloader.dataset
    raise KeyError('Config must define train_dataloader or test_dataloader.')


def get_validation_dataset_cfg(cfg):
    if 'validation_dataloader' in cfg:
        return cfg.validation_dataloader.dataset
    if 'val_dataloader' in cfg:
        return cfg.val_dataloader.dataset
    return None


def build_dataset(dataset_cfg):
    return LightweightSegDataset(dataset_cfg)


def save_calibration_checkpoint(model, path, epoch, train_loss, val_loss,
                                best_epoch=None, best_loss=None):
    checkpoint = {
        'prompt_score_version': model.PROMPT_SCORE_VERSION,
        'prompt_parameterization': model.prompt_parameterization,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'semantic_weight': model.semantic_weight.detach().cpu(),
        'presence_gain': model.presence_gain.detach().cpu(),
        'presence_bias': model.presence_bias.detach().cpu(),
        'log_prompt_weight': model.log_prompt_weight.detach().cpu(),
        'prompt_weight': model.prompt_weight.detach().cpu(),
        'prompt_threshold': model.prompt_threshold.detach().cpu(),
        'min_prompt_component_area': model.min_prompt_component_area.detach().cpu(),
        'concept_threshold': model.concept_threshold.detach().cpu(),
        'concept_weight': model.concept_weight.detach().cpu(),
        'query_words': model.query_words,
        'query_idx': model.query_idx.detach().cpu(),
        'concepts': model.concepts,
        'trainable_parameter_scope': 'per_prompt_only',
        'trainable_prompt_parameter_names': list(
            model.TRAINABLE_PROMPT_PARAMETER_NAMES),
    }
    for name, values in model.prompt_parameter_items():
        checkpoint[name] = values.detach().cpu()
    torch.save(checkpoint, path)


def log_calibration_parameters(writer, model, step):
    if writer is None:
        return
    for name, tensor in model.prompt_parameter_items():
        values = tensor.detach().float().cpu()
        writer.add_scalar(f'params/{name}_mean', values.mean().item(), step)
        writer.add_scalar(f'params/{name}_min', values.min().item(), step)
        writer.add_scalar(f'params/{name}_max', values.max().item(), step)
        writer.add_histogram(f'params_hist/{name}', values, step)
    prompt_weight = model.prompt_weight.detach().float().cpu()
    writer.add_scalar(
        'params/prompt_weight_mean', prompt_weight.mean().item(), step)
    writer.add_scalar(
        'params/prompt_weight_min', prompt_weight.min().item(), step)
    writer.add_scalar(
        'params/prompt_weight_max', prompt_weight.max().item(), step)
    writer.add_histogram('params_hist/prompt_weight', prompt_weight, step)


def calibration_regularization(model):
    terms = [
        (parameter - model.initial_prompt_parameter(name)).pow(2).mean()
        for name, parameter in model.prompt_parameter_items()
    ]
    return sum(terms)


def constrain_calibration_parameters(model, args):
    model.constrain_prompt_parameters(
        presence_gain_max=args.presence_gain_max,
        prompt_weight_max=args.prompt_weight_max)


def _fit_kmeans(features, num_clusters, num_starts=16, max_iters=100):
    num_samples = features.shape[0]
    best_result = None
    for seed in range(num_starts):
        rng = np.random.default_rng(seed)
        centers = features[
            rng.choice(num_samples, size=num_clusters, replace=False)].copy()
        labels = None
        for _ in range(max_iters):
            squared_distances = ((
                features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            next_labels = squared_distances.argmin(axis=1)
            if labels is not None and np.array_equal(labels, next_labels):
                break
            labels = next_labels
            min_distances = squared_distances.min(axis=1)
            next_centers = centers.copy()
            for cluster_idx in range(num_clusters):
                members = features[labels == cluster_idx]
                if len(members) > 0:
                    next_centers[cluster_idx] = members.mean(axis=0)
                else:
                    next_centers[cluster_idx] = features[
                        int(np.argmax(min_distances))]
            centers = next_centers

        squared_distances = ((
            features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = squared_distances.argmin(axis=1)
        if len(np.unique(labels)) != num_clusters:
            continue
        inertia = float(squared_distances[np.arange(num_samples), labels].sum())
        if best_result is None or inertia < best_result[2]:
            best_result = labels.copy(), centers.copy(), inertia
    return best_result


def _silhouette_score(features, labels):
    cluster_ids = np.unique(labels)
    if len(cluster_ids) < 2:
        return float('nan')
    distances = np.sqrt(((
        features[:, None, :] - features[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for sample_idx, cluster_idx in enumerate(labels):
        same_cluster = np.flatnonzero(labels == cluster_idx)
        if len(same_cluster) <= 1:
            scores.append(0.0)
            continue
        within_distance = (
            distances[sample_idx, same_cluster].sum()
            / (len(same_cluster) - 1))
        nearest_cluster_distance = min(
            distances[sample_idx, labels == other_idx].mean()
            for other_idx in cluster_ids
            if other_idx != cluster_idx)
        denominator = max(within_distance, nearest_cluster_distance)
        scores.append(
            0.0 if denominator <= 1e-12 else
            (nearest_cluster_distance - within_distance) / denominator)
    return float(np.mean(scores))


def analyze_prompt_parameter_clusters(model, output_path, records=None,
                                      min_hard_iou_gain=0.0,
                                      min_cluster_size=2,
                                      max_clusters=8):
    """Cluster interpretable behavior vectors for successful prompts only."""
    variation_floor = 1e-4
    optimization_parameter_items = model.prompt_parameter_items()
    semantic_weight = model.semantic_weight.detach().float().cpu().numpy()
    presence_gain = model.presence_gain.detach().float().cpu().numpy()
    presence_bias = model.presence_bias.detach().float().cpu().numpy()
    log_prompt_weight = model.log_prompt_weight.detach().float().cpu().numpy()
    presence_intercept = (
        log_prompt_weight + presence_gain * (0.5 - presence_bias))
    parameter_names = [
        'semantic_weight', 'presence_gain', 'presence_intercept']
    parameter_matrix_all = np.stack([
        semantic_weight, presence_gain, presence_intercept], axis=1)

    records_by_index = {
        int(record['prompt_index']): record for record in (records or [])
    }
    if records is None:
        selected_indices = list(range(model.num_queries))
    else:
        selected_indices = [
            index for index, record in records_by_index.items()
            if (record.get('prompt_exit') == 'prototype'
                and float(record.get('hard_iou_gain_vs_epoch0', 0.0))
                > min_hard_iou_gain
                and float(record.get('selected_metrics', {}).get(
                    'recall') or 0.0) > 0.0)
        ]
    selected_indices.sort()
    parameter_matrix = parameter_matrix_all[selected_indices]

    behavior_feature_names = list(parameter_names)
    behavior_rows = []
    for prompt_idx in selected_indices:
        row = list(parameter_matrix_all[prompt_idx])
        if records is not None:
            record = records_by_index[prompt_idx]
            baseline = record['baseline_metrics']
            selected = record['selected_metrics']
            behavior_feature_names = [
                *parameter_names,
                'hard_iou_gain_vs_epoch0',
                'precision_change',
                'recall_change',
                'pred_gt_area_ratio_change',
            ]
            row.extend([
                float(record['hard_iou_gain_vs_epoch0']),
                float((selected.get('precision') or 0.0)
                      - (baseline.get('precision') or 0.0)),
                float((selected.get('recall') or 0.0)
                      - (baseline.get('recall') or 0.0)),
                float((selected.get('pred_gt_area_ratio') or 0.0)
                      - (baseline.get('pred_gt_area_ratio') or 0.0)),
            ])
        behavior_rows.append(row)
    clustering_matrix = np.asarray(behavior_rows, dtype=np.float64)
    if clustering_matrix.size == 0:
        clustering_matrix = np.zeros(
            (0, len(behavior_feature_names)), dtype=np.float64)

    means = (
        clustering_matrix.mean(axis=0)
        if len(clustering_matrix) else np.zeros(len(behavior_feature_names)))
    stds = (
        clustering_matrix.std(axis=0)
        if len(clustering_matrix) else np.zeros(len(behavior_feature_names)))
    variable_mask = stds > variation_floor
    normalized = (
        (clustering_matrix[:, variable_mask] - means[variable_mask])
        / stds[variable_mask]
        if np.any(variable_mask)
        else np.zeros((len(clustering_matrix), 0), dtype=np.float32))

    candidate_scores = []
    best_clustering = None

    def evaluate_range(last_k):
        nonlocal best_clustering
        evaluated = {item['num_clusters'] for item in candidate_scores}
        for num_clusters in range(2, last_k + 1):
            if num_clusters in evaluated:
                continue
            result = _fit_kmeans(normalized, num_clusters)
            if result is None:
                continue
            labels, centers, inertia = result
            cluster_sizes = np.bincount(labels, minlength=num_clusters)
            if np.any(cluster_sizes < min_cluster_size):
                continue
            silhouette = _silhouette_score(normalized, labels)
            gains = (
                np.asarray([
                    records_by_index[index]['hard_iou_gain_vs_epoch0']
                    for index in selected_indices], dtype=np.float64)
                if records is not None else np.zeros(len(labels)))
            candidate_scores.append(dict(
                num_clusters=num_clusters,
                silhouette_score=silhouette,
                inertia=inertia,
                cluster_sizes=[int(size) for size in cluster_sizes],
                cluster_mean_hard_iou_gain=[
                    float(gains[labels == cluster_idx].mean())
                    for cluster_idx in range(num_clusters)],
                successful_prompt_ratio=1.0))
            if (best_clustering is None
                    or silhouette > best_clustering['silhouette_score']):
                best_clustering = dict(
                    labels=labels,
                    centers=centers,
                    num_clusters=num_clusters,
                    silhouette_score=silhouette,
                    inertia=inertia)

    feasible_max = len(clustering_matrix) // min_cluster_size
    initial_max = min(max_clusters, feasible_max)
    if normalized.shape[1] > 0 and initial_max >= 2:
        evaluate_range(initial_max)
    upper_bound_extended = False
    upper_bound_converged = True
    if (best_clustering is not None
            and best_clustering['num_clusters'] == initial_max
            and feasible_max > initial_max):
        upper_bound_extended = True
        extended_max = min(feasible_max, max(initial_max + 1, initial_max * 2))
        evaluate_range(extended_max)
        upper_bound_converged = (
            best_clustering['num_clusters'] < extended_max)

    if best_clustering is None:
        labels = np.zeros(len(clustering_matrix), dtype=np.int64)
        num_clusters = 1 if len(clustering_matrix) else 0
        silhouette = None
        clusterability = 'none'
    else:
        labels = best_clustering['labels']
        num_clusters = best_clustering['num_clusters']
        silhouette = best_clustering['silhouette_score']
        clusterability = (
            'strong' if silhouette >= 0.5 else
            'moderate' if silhouette >= 0.25 else 'weak')

    prompt_keys = model._prompt_keys()
    query_indices = model.query_idx.detach().cpu().tolist()
    prompt_records = []
    for local_idx, prompt_idx in enumerate(selected_indices):
        prompt, concept_id, prompt_position = prompt_keys[prompt_idx]
        concept_idx = query_indices[prompt_idx]
        concept = model.concepts[concept_idx] if model.concepts else {}
        prompt_records.append(dict(
            prompt_index=prompt_idx,
            prompt=prompt,
            prompt_position=prompt_position,
            concept_index=concept_idx,
            concept_id=concept_id,
            concept_name=concept.get('concept_name', concept_id),
            cluster=int(labels[local_idx]) if len(labels) else None,
            parameters={
                name: float(parameter_matrix_all[prompt_idx, param_idx])
                for param_idx, name in enumerate(parameter_names)
            },
            clustering_features={
                name: float(clustering_matrix[local_idx, feature_idx])
                for feature_idx, name in enumerate(behavior_feature_names)
            }))

    clusters = []
    for cluster_idx in range(num_clusters):
        member_local_indices = np.flatnonzero(labels == cluster_idx)
        member_prompt_indices = [
            selected_indices[index] for index in member_local_indices]
        clusters.append(dict(
            cluster=cluster_idx,
            size=int(len(member_local_indices)),
            centroid={
                name: float(parameter_matrix[
                    member_local_indices, param_idx].mean())
                for param_idx, name in enumerate(parameter_names)
            },
            feature_centroid={
                name: float(clustering_matrix[
                    member_local_indices, feature_idx].mean())
                for feature_idx, name in enumerate(behavior_feature_names)
            },
            mean_hard_iou_gain=(
                float(np.mean([
                    records_by_index[index]['hard_iou_gain_vs_epoch0']
                    for index in member_prompt_indices]))
                if records is not None else None),
            prompt_indices=member_prompt_indices))

    excluded = {
        exit_name: [
            int(record['prompt_index']) for record in (records or [])
            if record.get('prompt_exit') == exit_name
        ]
        for exit_name in ('identity_fallback', 'reject')
    }
    analysis = dict(
        trainable_parameter_scope='per_prompt_only',
        parameter_names=parameter_names,
        optimization_parameter_names=[
            name for name, _ in optimization_parameter_items],
        clustering_parameterization='identifiable_behavior',
        clustering_feature_names=behavior_feature_names,
        prototype_filter=dict(
            min_hard_iou_gain=min_hard_iou_gain,
            min_cluster_size=min_cluster_size,
            num_input_prompts=len(selected_indices)),
        excluded_prompts=excluded,
        variation_floor=variation_floor,
        variable_parameter_names=[
            name for name, is_variable in zip(
                behavior_feature_names, variable_mask) if is_variable],
        normalization=dict(
            mean={name: float(value) for name, value in zip(
                behavior_feature_names, means)},
            std={name: float(value) for name, value in zip(
                behavior_feature_names, stds)}),
        selected_cluster_count=num_clusters,
        silhouette_score=silhouette,
        clusterability=clusterability,
        upper_bound_extended=upper_bound_extended,
        upper_bound_converged=upper_bound_converged,
        candidate_scores=candidate_scores,
        clusters=clusters,
        prompts=prompt_records)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    rows = []
    for record in sorted(prompt_records, key=lambda item: (
            item['cluster'], item['concept_index'], item['prompt_index'])):
        rows.append([
            record['cluster'],
            record['concept_id'],
            record['prompt'],
            *[f"{record['parameters'][name]:.4f}" for name in parameter_names],
        ])
    summary_path = osp.splitext(output_path)[0] + '.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(
            f'clusterability={clusterability} clusters={num_clusters} '
            f'silhouette={silhouette}\n')
        f.write(make_table(
            ['cluster', 'concept', 'prompt', *parameter_names], rows))
        f.write('\n')
    print(
        f'Saved prompt parameter clustering to {output_path} '
        f'(clusterability={clusterability}, clusters={num_clusters}, '
        f'silhouette={silhouette})')
    return analysis


def build_validation_metric_state(model):
    concepts = model.concepts or []
    class_names = [
        concept.get('concept_name') or concept.get('concept_id') or str(index)
        for index, concept in enumerate(concepts)
    ]
    # Plain prompt TXT files have no TNLS mapping.  Their concept order maps
    # directly to zero-based GT after reduce_zero_label=True (LoveDA).
    label_sets = []
    for concept_idx, concept in enumerate(concepts):
        labels = concept.get('tnls_dl3', [])
        label_sets.append(
            set(int(label) for label in labels) if labels else {concept_idx})
    prompt_metadata = []
    for concept_idx, concept in enumerate(concepts):
        prompts = concept.get('prompts', [])
        if isinstance(prompts, str):
            prompts = [prompts]
        for prompt in prompts:
            prompt_metadata.append(dict(
                prompt=prompt,
                concept_idx=concept_idx,
                concept_id=concept.get('concept_id'),
                concept_name=concept.get('concept_name')
                or concept.get('concept_id') or str(concept_idx)))

    num_classes = len(class_names)
    num_prompts = len(prompt_metadata)
    return dict(
        class_names=class_names,
        label_sets=label_sets,
        prompt_metadata=prompt_metadata,
        class_intersect=torch.zeros(num_classes, dtype=torch.float64),
        class_union=torch.zeros(num_classes, dtype=torch.float64),
        class_pred=torch.zeros(num_classes, dtype=torch.float64),
        class_label=torch.zeros(num_classes, dtype=torch.float64),
        prompt_intersect=torch.zeros(num_prompts, dtype=torch.float64),
        prompt_union=torch.zeros(num_prompts, dtype=torch.float64),
        prompt_pred=torch.zeros(num_prompts, dtype=torch.float64),
        prompt_label=torch.zeros(num_prompts, dtype=torch.float64))


def update_validation_metric_state(state,
                                   class_scores,
                                   prompt_scores,
                                   raw_label,
                                   ignore_index,
                                   score_threshold):
    label = raw_label.squeeze().to(class_scores.device)
    valid_mask = label != ignore_index
    for concept_idx, valid_labels in enumerate(state['label_sets']):
        pred_mask = class_scores[concept_idx][valid_mask] > score_threshold
        label_values = label[valid_mask]
        label_mask = torch.zeros_like(label_values, dtype=torch.bool)
        for valid_label in valid_labels:
            label_mask |= label_values == valid_label

        intersect = pred_mask & label_mask
        union = pred_mask | label_mask
        state['class_intersect'][concept_idx] += intersect.sum().detach().cpu()
        state['class_union'][concept_idx] += union.sum().detach().cpu()
        state['class_pred'][concept_idx] += pred_mask.sum().detach().cpu()
        state['class_label'][concept_idx] += label_mask.sum().detach().cpu()

    if prompt_scores is None:
        return
    prompt_scores = prompt_scores.to(class_scores.device)
    for prompt_idx, metadata in enumerate(state['prompt_metadata']):
        valid_labels = state['label_sets'][metadata['concept_idx']]
        pred_mask = prompt_scores[prompt_idx][valid_mask] > score_threshold
        label_values = label[valid_mask]
        label_mask = torch.zeros_like(label_values, dtype=torch.bool)
        for valid_label in valid_labels:
            label_mask |= label_values == valid_label

        intersect = pred_mask & label_mask
        union = pred_mask | label_mask
        state['prompt_intersect'][prompt_idx] += intersect.sum().detach().cpu()
        state['prompt_union'][prompt_idx] += union.sum().detach().cpu()
        state['prompt_pred'][prompt_idx] += pred_mask.sum().detach().cpu()
        state['prompt_label'][prompt_idx] += label_mask.sum().detach().cpu()


def metric_rows(intersect, union, pred, label, names):
    iou = safe_divide(intersect, union)
    acc = safe_divide(intersect, label)
    precision = safe_divide(intersect, pred)
    recall = acc
    fscore = safe_divide(2 * precision * recall, precision + recall)
    rows = []
    for index, name in enumerate(names):
        rows.append([
            name,
            format_metric_value(iou[index]),
            format_metric_value(acc[index]),
            format_metric_value(fscore[index]),
            format_metric_value(precision[index]),
            format_metric_value(recall[index])
        ])
    return rows


def metric_summary(intersect, union, pred, label):
    iou = safe_divide(intersect, union)
    acc = safe_divide(intersect, label)
    recall = acc

    def nanmean_percent(values):
        valid = ~torch.isnan(values)
        if not torch.any(valid):
            return float('nan')
        return float((values[valid].mean() * 100).item())

    return dict(
        mIoU=nanmean_percent(iou),
        mAcc=nanmean_percent(acc),
        mRecall=nanmean_percent(recall))


def validation_report(state):
    headers = ['Class', 'IoU', 'Acc', 'Fscore', 'Precision', 'Recall']
    summary = metric_summary(
        state['class_intersect'],
        state['class_union'],
        state['class_pred'],
        state['class_label'])
    class_rows = metric_rows(
        state['class_intersect'],
        state['class_union'],
        state['class_pred'],
        state['class_label'],
        state['class_names'])
    class_table = make_table(headers, class_rows)

    prompt_names = [
        f"{metadata['concept_id'] or metadata['concept_idx']}::{metadata['prompt']}"
        for metadata in state['prompt_metadata']
    ]
    prompt_rows = metric_rows(
        state['prompt_intersect'],
        state['prompt_union'],
        state['prompt_pred'],
        state['prompt_label'],
        prompt_names)
    prompt_table = make_table(
        ['Prompt', 'IoU', 'Acc', 'Fscore', 'Precision', 'Recall'],
        prompt_rows)
    return class_table, prompt_table, summary


def validation_metric_records(state, epoch=None, split='validation',
                              threshold=None):
    """Return shared-workbook rows from the lightweight metric accumulator."""
    class_rows = []
    for index, name in enumerate(state['class_names']):
        row = dict(Class=name, Epoch=epoch, Split=split, Threshold=threshold)
        row.update(MetricsReporter.counts_to_metrics(
            state['class_intersect'][index], state['class_union'][index],
            state['class_pred'][index], state['class_label'][index]))
        class_rows.append(row)
    prompt_rows = []
    for index, metadata in enumerate(state['prompt_metadata']):
        row = dict(
            Prompt=metadata['prompt'],
            ConceptID=metadata['concept_id'],
            ConceptName=metadata['concept_name'],
            Epoch=epoch,
            Split=split,
            Threshold=threshold)
        row.update(MetricsReporter.counts_to_metrics(
            state['prompt_intersect'][index], state['prompt_union'][index],
            state['prompt_pred'][index], state['prompt_label'][index]))
        prompt_rows.append(row)
    return class_rows, prompt_rows


def build_label_map(concepts, ignore_index):
    if not concepts:
        return None
    max_label = -1
    for concept in concepts:
        for label in concept.get('tnls_dl3', []):
            max_label = max(max_label, int(label))
    if max_label < 0:
        return None

    label_map = torch.full((max_label + 1,), int(ignore_index), dtype=torch.long)
    for concept_idx, concept in enumerate(concepts):
        for label in concept.get('tnls_dl3', []):
            label_map[int(label)] = concept_idx
    return label_map


def reduce_metric_loss(loss_values, valid_mask, worst_weight):
    valid_mask = valid_mask & torch.isfinite(loss_values)
    if not torch.any(valid_mask):
        return loss_values.new_tensor(0.0)
    selected = loss_values[valid_mask]
    loss = selected.mean()
    if worst_weight > 0:
        loss = loss + worst_weight * selected.max()
    return loss


def soft_concept_metric_loss(class_scores, gt, ignore_index, args):
    num_classes = class_scores.shape[0]
    valid_mask = (
        (gt != ignore_index) & (gt >= 0) & (gt < num_classes))
    if not torch.any(valid_mask):
        zero = class_scores.sum() * 0.0
        return zero, zero

    probs = torch.softmax(class_scores.float(), dim=0)
    probs_flat = probs[:, valid_mask]
    gt_flat = gt[valid_mask].long()
    target = F.one_hot(gt_flat, num_classes=num_classes)
    target = target.transpose(0, 1).to(dtype=probs_flat.dtype)

    eps = float(args.soft_loss_eps)
    intersect = (probs_flat * target).sum(dim=1)
    pred_mass = probs_flat.sum(dim=1)
    label_mass = target.sum(dim=1)
    union = pred_mass + label_mass - intersect
    present_mask = label_mass > 0

    iou = (intersect + eps) / (union + eps)
    iou_losses = 1.0 - iou
    iou_mask = present_mask if args.soft_iou_present_only else torch.ones_like(
        present_mask, dtype=torch.bool)
    soft_iou_loss = reduce_metric_loss(
        iou_losses, iou_mask, args.worst_concept_loss_weight)

    recall = (intersect + eps) / (label_mass + eps)
    recall_losses = F.relu(float(args.recall_target) - recall)
    recall_loss = reduce_metric_loss(
        recall_losses, present_mask, args.worst_concept_loss_weight)
    return soft_iou_loss, recall_loss


def compute_supervised_loss(class_scores, gt, ignore_index, args):
    total_loss = class_scores.sum() * 0.0
    has_loss = False

    if args.ce_loss_weight > 0:
        ce_loss = F.cross_entropy(
            class_scores.unsqueeze(0).float(),
            gt.unsqueeze(0),
            ignore_index=ignore_index)
        total_loss = total_loss + args.ce_loss_weight * ce_loss
        has_loss = True

    if args.soft_iou_loss_weight > 0 or args.recall_loss_weight > 0:
        soft_iou_loss, recall_loss = soft_concept_metric_loss(
            class_scores, gt, ignore_index, args)
        if args.soft_iou_loss_weight > 0:
            total_loss = total_loss + args.soft_iou_loss_weight * soft_iou_loss
            has_loss = True
        if args.recall_loss_weight > 0:
            total_loss = total_loss + args.recall_loss_weight * recall_loss
            has_loss = True

    if not has_loss:
        raise ValueError(
            'At least one supervised loss weight must be positive. Check '
            '--ce-loss-weight, --soft-iou-loss-weight, and '
            '--recall-loss-weight.')
    return total_loss


def compute_sample_loss(model,
                        sample,
                        device,
                        ignore_index,
                        label_map=None,
                        return_scores=False,
                        args=None):
    data_sample = sample['data_samples']
    image_path = data_sample.metainfo.get('img_path')
    image = Image.open(image_path).convert('RGB')

    gt = get_gt(
        data_sample,
        device,
        label_map=label_map,
        ignore_index=ignore_index)
    if not torch.any(gt != ignore_index):
        return (None, None, None) if return_scores else None
    class_scores, raw_prompt_scores = model.score_image(image, ori_shape=gt.shape)
    class_scores = resize_scores(class_scores, gt.shape)

    if args is None:
        args = SimpleNamespace(
            ce_loss_weight=1.0,
            soft_iou_loss_weight=0.0,
            recall_loss_weight=0.0,
            recall_target=0.3,
            worst_concept_loss_weight=0.0,
            soft_iou_present_only=False,
            soft_loss_eps=1e-6)
    loss = compute_supervised_loss(class_scores, gt, ignore_index, args)
    if return_scores:
        return loss, class_scores, raw_prompt_scores
    return loss


def train_one_epoch(model, dataset, optimizer, device, ignore_index, epoch,
                    global_iter, max_iters, label_map=None, writer=None,
                    progress_interval=10, total_epochs=None, args=None):
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    num_samples = 0
    epoch_start = time.time()
    target_samples = len(dataset)
    if max_iters > 0:
        target_samples = min(target_samples, max_iters - global_iter)
    target_samples = max(target_samples, 1)

    for index in range(len(dataset)):
        loss = compute_sample_loss(
            model,
            dataset[index],
            device,
            ignore_index,
            label_map=label_map,
            args=args)
        if loss is None:
            continue

        reg_loss = calibration_regularization(model)
        total_step_loss = loss
        if args is not None and args.param_reg_weight > 0:
            total_step_loss = loss + args.param_reg_weight * reg_loss

        optimizer.zero_grad()
        total_step_loss.backward()
        if args is not None and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.prompt_parameters(), args.grad_clip_norm)
        optimizer.step()
        if args is not None:
            constrain_calibration_parameters(model, args)

        loss_value = float(loss.detach().cpu())
        total_loss += loss_value
        running_loss += loss_value
        num_samples += 1
        global_iter += 1

        should_print = (
            progress_interval > 0 and num_samples % progress_interval == 0)
        if should_print:
            avg_loss = running_loss / max(progress_interval, 1)
            elapsed = time.time() - epoch_start
            print(progress_line(
                'train',
                epoch,
                num_samples,
                target_samples,
                avg_loss,
                elapsed,
                total_epochs=total_epochs),
                  flush=True)
            if writer is not None:
                writer.add_scalar('loss/train_iter', avg_loss, global_iter)
            running_loss = 0.0
        if max_iters > 0 and global_iter >= max_iters:
            break

    return total_loss / max(num_samples, 1), global_iter


@torch.no_grad()
def validate(model,
             dataset,
             device,
             ignore_index,
             label_map=None,
             epoch=None,
             vis_dir=None,
             max_samples=-1,
             progress_interval=10,
             total_epochs=None,
             log_file=None,
             args=None):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    val_start = time.time()
    target_samples = len(dataset)
    if max_samples > 0:
        target_samples = min(target_samples, max_samples)
    metric_state = build_validation_metric_state(model)

    for index in range(len(dataset)):
        if max_samples > 0 and num_samples >= max_samples:
            break
        sample = dataset[index]
        loss, class_scores, raw_prompt_scores = compute_sample_loss(
            model,
            sample,
            device,
            ignore_index,
            label_map=label_map,
            return_scores=True,
            args=args)
        if loss is None:
            continue
        if vis_dir is not None and epoch is not None:
            save_validation_visualization(
                model,
                sample,
                class_scores,
                vis_dir,
                epoch,
                ignore_index=ignore_index)
        update_validation_metric_state(
            metric_state,
            class_scores,
            raw_prompt_scores,
            sample['data_samples'].gt_sem_seg.data,
            ignore_index,
            model.prob_thd)
        total_loss += float(loss.detach().cpu())
        num_samples += 1
        if progress_interval > 0 and num_samples % progress_interval == 0:
            avg_loss = total_loss / max(num_samples, 1)
            elapsed = time.time() - val_start
            print(progress_line(
                'val',
                epoch,
                num_samples,
                target_samples,
                avg_loss,
                elapsed,
                total_epochs=total_epochs),
                  flush=True)

    val_loss = total_loss / max(num_samples, 1)
    class_table, prompt_table, summary = validation_report(metric_state)
    report_header = (
        f'\nValidation epoch={epoch} samples={num_samples} '
        f'loss={val_loss:.6f}\n'
        f"mIoU={summary['mIoU']:.2f} "
        f"mAcc={summary['mAcc']:.2f} "
        f"mRecall={summary['mRecall']:.2f}\n")
    print(report_header + class_table, flush=True)
    if log_file is not None:
        os.makedirs(osp.dirname(log_file), exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(report_header)
            f.write(class_table)
            f.write('\n\nPrompt IoU results:\n')
            f.write(prompt_table)
            f.write('\n\n')
    if args is not None and getattr(args, 'save_dir', None):
        class_rows, prompt_rows = validation_metric_records(
            metric_state,
            epoch=epoch,
            split='validation',
            threshold=model.prob_thd)
        summary_row = dict(
            mIoU=summary['mIoU'],
            mAcc=summary['mAcc'],
            mRecall=summary['mRecall'],
            Epoch=epoch,
            Split='validation',
            Loss=val_loss)
        MetricsReporter.write_workbook(
            osp.join(args.save_dir, 'results.xlsx'),
            summaries=[summary_row],
            class_metrics=class_rows,
            prompt_metrics=prompt_rows,
            model=getattr(model, 'model_type', model.__class__.__name__),
            dataset=osp.splitext(osp.basename(args.config))[0])
    return val_loss, summary


def main():
    args = parse_args()
    cfg = load_config(args.config)
    merge_cfg_options(cfg, args.cfg_options)
    apply_learnable_train_config(args, cfg)

    os.makedirs(args.save_dir, exist_ok=True)
    save_experiment_snapshot(args, cfg)
    tensorboard_dir = osp.join(args.save_dir, 'tensorboard')
    writer = build_summary_writer(tensorboard_dir)
    print(describe_loss_args(args))
    model = build_model(cfg.model)
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    prompt_parameters = model.prompt_parameters()
    for name, param in model.prompt_parameter_items():
        if param.shape != (model.num_queries,):
            raise ValueError(
                f'Prompt parameter {name} must have one independent value per '
                f'prompt: expected {(model.num_queries,)}, got {param.shape}.')
    for param in prompt_parameters:
        param.requires_grad_(True)
    trainable_names = [
        name for name, param in model.named_parameters()
        if param.requires_grad]
    expected_names = {
        f'prompt_calibrations.prompts.{prompt_idx}.{name}'
        for prompt_idx in range(model.num_queries)
        for name in model.TRAINABLE_PROMPT_PARAMETER_NAMES
    }
    if set(trainable_names) != expected_names:
        raise RuntimeError(
            'Only per-prompt parameters may be trainable. '
            f'Expected {sorted(expected_names)}, got {sorted(trainable_names)}.')
    print(
        f'Training {model.num_queries} independent prompt groups '
        f'({len(prompt_parameters)} scalar parameters); concept parameters '
        'and SAM3 are frozen.')
    constrain_calibration_parameters(model, args)

    optimizer = torch.optim.Adam(prompt_parameters, lr=args.lr)
    train_dataset = build_dataset(get_train_dataset_cfg(cfg))
    validation_dataset_cfg = get_validation_dataset_cfg(cfg)
    validation_dataset = (
        build_dataset(validation_dataset_cfg)
        if validation_dataset_cfg is not None else None)
    ignore_index = cfg.test_evaluator.get('ignore_index', 255)
    device = model.device
    label_map = build_label_map(model.concepts, ignore_index)
    if label_map is not None:
        print('Using TNLS DL3 to concept-index label mapping for loss.')

    global_iter = 0
    best_epoch = None
    best_loss = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, global_iter = train_one_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            device=device,
            ignore_index=ignore_index,
            epoch=epoch,
            global_iter=global_iter,
            max_iters=args.max_iters,
            label_map=label_map,
            writer=writer,
            progress_interval=args.progress_interval,
            total_epochs=args.epochs,
            args=args)

        val_loss = None
        should_validate = (
            validation_dataset is not None
            and args.validate_every_epochs > 0
            and epoch % args.validate_every_epochs == 0)
        if should_validate:
            val_loss, val_summary = validate(
                model,
                validation_dataset,
                device,
                ignore_index,
                label_map=label_map,
                epoch=epoch,
                vis_dir=args.save_dir,
                max_samples=args.max_val_samples,
                progress_interval=args.progress_interval,
                total_epochs=args.epochs,
                log_file=osp.join(args.save_dir, 'validation_log.txt'),
                args=args)
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"mIoU={val_summary['mIoU']:.2f} "
                f"mAcc={val_summary['mAcc']:.2f} "
                f"mRecall={val_summary['mRecall']:.2f}")
        else:
            val_summary = None
            print(f'epoch={epoch} train_loss={train_loss:.6f}')

        has_validation = (
            validation_dataset is not None and args.validate_every_epochs > 0)
        monitor_loss = val_loss if has_validation else train_loss
        if writer is not None:
            writer.add_scalar('loss/train_epoch', train_loss, epoch)
            if val_loss is not None:
                writer.add_scalar('loss/validation_epoch', val_loss, epoch)
                if val_summary is not None:
                    writer.add_scalar(
                        'metrics/validation_mIoU', val_summary['mIoU'], epoch)
                    writer.add_scalar(
                        'metrics/validation_mAcc', val_summary['mAcc'], epoch)
                    writer.add_scalar(
                        'metrics/validation_mRecall',
                        val_summary['mRecall'],
                        epoch)
            if monitor_loss is not None:
                writer.add_scalar('loss/monitor', monitor_loss, epoch)
        if monitor_loss is not None and (
                best_loss is None or monitor_loss < best_loss):
            best_loss = monitor_loss
            best_epoch = epoch
            best_path = osp.join(args.save_dir, 'best_epoch.pth')
            save_calibration_checkpoint(
                model, best_path, epoch, train_loss, val_loss,
                best_epoch=best_epoch, best_loss=best_loss)
            analyze_prompt_parameter_clusters(
                model,
                osp.join(args.save_dir, 'best_prompt_parameter_clusters.json'))
            print(f'Saved best checkpoint to {best_path}')
        if writer is not None and best_loss is not None:
            writer.add_scalar('loss/best', best_loss, epoch)
            writer.add_scalar('checkpoint/best_epoch', best_epoch, epoch)
            log_calibration_parameters(writer, model, epoch)

        if (args.save_every_epochs > 0
                and epoch % args.save_every_epochs == 0):
            epoch_path = osp.join(args.save_dir, f'epoch_{epoch:04d}.pth')
            save_calibration_checkpoint(
                model, epoch_path, epoch, train_loss, val_loss,
                best_epoch=best_epoch, best_loss=best_loss)
            print(f'Saved periodic checkpoint to {epoch_path}')

        history.append(dict(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=best_epoch,
            best_loss=best_loss,
            loss_mode=args.loss_mode,
            ce_loss_weight=args.ce_loss_weight,
            soft_iou_loss_weight=args.soft_iou_loss_weight,
            recall_loss_weight=args.recall_loss_weight,
            recall_target=args.recall_target,
            worst_concept_loss_weight=args.worst_concept_loss_weight,
            soft_iou_present_only=args.soft_iou_present_only))
        with open(osp.join(args.save_dir, 'training_history.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        if args.max_iters > 0 and global_iter >= args.max_iters:
            break

    save_path = osp.join(args.save_dir, 'segearthov3_learnable_params.pth')
    final_epoch = history[-1]['epoch'] if history else 0
    final_train_loss = history[-1]['train_loss'] if history else None
    final_val_loss = history[-1]['val_loss'] if history else None
    save_calibration_checkpoint(
        model, save_path, final_epoch, final_train_loss, final_val_loss,
        best_epoch=best_epoch, best_loss=best_loss)
    analyze_prompt_parameter_clusters(
        model, osp.join(args.save_dir, 'prompt_parameter_clusters.json'))
    print(f'Saved learnable parameters to {save_path}')
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == '__main__':
    main()
