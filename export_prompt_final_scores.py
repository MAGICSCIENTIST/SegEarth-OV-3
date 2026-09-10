"""Export independently calibrated SAM3 prompt scores as continuous channels.

The script restores the model configuration from an independent-prompt training
run, extracts the three prompt heads for each input image, and rebuilds the
selected prompt's final score with the exact Stage A formula.
"""

import argparse
import json
import math
import os
import os.path as osp
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('SEGOV3_LIGHTWEIGHT_TRAIN', '1')

import numpy as np
import torch
from PIL import Image

from segearthov3_learnable_segmentor import (
    SCORE_EPS,
    SegEarthOV3LearnableSegmentation,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EXTENSIONS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
INVALID_WINDOWS_CHARS = '<>:"/\\|?*'
WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    *(f'COM{index}' for index in range(1, 10)),
    *(f'LPT{index}' for index in range(1, 10)),
}
FINAL_SCORE_FORMULA = (
    'fused = semantic_weight * semantic_score_map + '
    '(1 - semantic_weight) * instance_score_map; '
    'final_score = sigmoid(logit(clamp(fused, 1e-6, 1 - 1e-6)) + '
    'presence_gain * (presence_score - presence_bias) + log(prompt_weight))'
)


@dataclass
class PromptCheckpoint:
    path: Path
    checkpoint: dict
    record: dict

    @property
    def prompt_index(self):
        return int(self.record['prompt_index'])

    @property
    def prompt(self):
        return str(self.record['prompt'])

    @property
    def concept_id(self):
        return str(self.record['concept_id'])

    @property
    def concept_name(self):
        return str(self.record.get('concept_name', self.concept_id))

    @property
    def parameters(self):
        return self.record['parameters']

    @property
    def optimization_parameters(self):
        return self.checkpoint['parameters']

    @property
    def best_epoch(self):
        value = self.record.get(
            'best_epoch', self.record.get('selected_epoch'))
        if value is None:
            raise ValueError(
                f'Checkpoint has no best/selected epoch: {self.path}')
        return int(value)


class RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.count += int(values.size)
        self.total += float(values.sum())
        self.total_square += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def as_dict(self):
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            'count': self.count,
            'min': self.minimum,
            'max': self.maximum,
            'mean': mean,
            'std': math.sqrt(variance),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export selected SAM3 prompt final_score channels.')
    parser.add_argument(
        '--best-params-dir', required=True,
        help='Training result root or its prompt_best_params directory.')
    parser.add_argument(
        '--data-dir', required=True,
        help='Input image directory. It is not interpreted as train or val.')
    parser.add_argument(
        '--output-dir', required=True,
        help='Output root: <root>/<prompt name>/<image stem>.npy.')
    parser.add_argument(
        '--vis-folder', default=None,
        help=(
            'Optional directory for best-epoch diagnostic panels. No panels '
            'or visualization directory are created when omitted.'))
    parser.add_argument(
        '--prompts', nargs='*', required=True,
        help=(
            'Prompt names to export. Quote names containing spaces. For a '
            'duplicate name use "prompt@concept_id".'))
    parser.add_argument(
        '--config', default=None,
        help='Optional config fallback/override when no run snapshot exists.')
    parser.add_argument(
        '--concept-path', default=None,
        help='Optional concept JSON override. The run snapshot copy is default.')
    parser.add_argument(
        '--extensions', nargs='*', default=list(DEFAULT_EXTENSIONS),
        help='Input extensions, with or without a leading dot.')
    parser.add_argument(
        '--recursive', action='store_true',
        help='Recursively search data-dir. Duplicate image stems are rejected.')
    parser.add_argument(
        '--dtype', choices=('float16', 'float32'), default='float16',
        help='Saved NPY dtype. Computation is always float32.')
    parser.add_argument(
        '--device', default='cuda',
        help='CUDA device, for example cuda or cuda:1.')
    parser.add_argument(
        '--verify-atol', type=float, default=2e-6,
        help='Maximum formula-vs-model absolute error allowed per score map.')
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='Optional smoke-test limit after deterministic filename sorting.')
    parser.add_argument(
        '--progress-interval', type=int, default=10,
        help='Print progress every N images.')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Recompute and replace existing score files and panels.')
    args = parser.parse_args()

    if not args.prompts:
        parser.error('--prompts must contain at least one prompt name.')
    if args.max_images is not None and args.max_images <= 0:
        parser.error('--max-images must be greater than zero.')
    if args.progress_interval <= 0:
        parser.error('--progress-interval must be greater than zero.')
    if args.verify_atol < 0:
        parser.error('--verify-atol cannot be negative.')
    return args


def load_torch_file(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def resolve_best_params_dir(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Best-parameter directory not found: {path}')
    child = path / 'prompt_best_params'
    if child.is_dir() and any(child.glob('*.pth')):
        return child
    if path.name.casefold() == 'prompt_best_params' and any(path.glob('*.pth')):
        return path
    candidates = [
        item for item in path.rglob('prompt_best_params')
        if item.is_dir() and any(item.glob('*.pth'))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        formatted = '\n'.join(f'  {item}' for item in candidates)
        raise ValueError(
            'Multiple prompt_best_params directories were found; pass one '
            f'directly:\n{formatted}')
    raise FileNotFoundError(
        f'No prompt_best_params checkpoints were found under: {path}')


def load_prompt_checkpoints(best_params_dir):
    checkpoints = []
    seen_indices = {}
    for path in sorted(best_params_dir.glob('*.pth')):
        checkpoint = load_torch_file(path)
        record = checkpoint.get('record')
        if not isinstance(record, dict):
            raise ValueError(f'Checkpoint has no record object: {path}')
        required_record = {
            'prompt_index', 'prompt', 'concept_id', 'parameters',
        }
        missing = required_record - set(record)
        if missing:
            raise ValueError(
                f'Checkpoint {path} misses record fields: {sorted(missing)}')
        if not isinstance(checkpoint.get('parameters'), dict):
            raise ValueError(
                f'Checkpoint has no selected optimization parameters: {path}')
        item = PromptCheckpoint(path, checkpoint, record)
        if item.prompt_index in seen_indices:
            raise ValueError(
                f'Duplicate prompt index {item.prompt_index} in {path} and '
                f'{seen_indices[item.prompt_index]}')
        seen_indices[item.prompt_index] = path
        checkpoints.append(item)
    if not checkpoints:
        raise FileNotFoundError(f'No .pth files found in {best_params_dir}')
    return checkpoints


def normalize_key(value):
    return str(value).strip().casefold()


def resolve_prompt_selection(selectors, checkpoints):
    selected = []
    for selector in selectors:
        prompt_part = selector
        concept_part = None
        if '@' in selector:
            prompt_part, concept_part = selector.rsplit('@', 1)
        prompt_matches = [
            item for item in checkpoints
            if normalize_key(item.prompt) == normalize_key(prompt_part)
        ]
        if concept_part is not None:
            prompt_matches = [
                item for item in prompt_matches
                if normalize_key(concept_part) in {
                    normalize_key(item.concept_id),
                    normalize_key(item.concept_name),
                }
            ]
        if not prompt_matches:
            available = ', '.join(
                f'{item.prompt}@{item.concept_id}' for item in checkpoints)
            raise ValueError(
                f'Prompt selector {selector!r} did not match. Available: '
                f'{available}')
        if len(prompt_matches) > 1:
            choices = ', '.join(
                f'{item.prompt}@{item.concept_id}' for item in prompt_matches)
            raise ValueError(
                f'Prompt name {selector!r} is ambiguous. Use one of: '
                f'{choices}')
        selected.append(prompt_matches[0])

    indices = [item.prompt_index for item in selected]
    if len(indices) != len(set(indices)):
        raise ValueError('The same prompt checkpoint was selected more than once.')
    return selected


def safe_prompt_folder(prompt):
    name = ''.join(
        '_' if char in INVALID_WINDOWS_CHARS or ord(char) < 32 else char
        for char in str(prompt)).strip(' .')
    if not name:
        raise ValueError(f'Prompt has no usable folder name: {prompt!r}')
    if name.upper() in WINDOWS_RESERVED_NAMES:
        name = f'_{name}'
    return name


def find_run_snapshot(best_params_dir):
    search_roots = [best_params_dir.parent, best_params_dir]
    search_roots.extend(list(best_params_dir.parents)[:3])
    visited = set()
    for root in search_roots:
        root = root.resolve()
        if root in visited:
            continue
        visited.add(root)
        snapshot_path = root / 'config_snapshot.yaml'
        if snapshot_path.is_file():
            with snapshot_path.open('r', encoding='utf-8') as file:
                snapshot = json.load(file)
            return root, snapshot_path, snapshot
    return None, None, None


def resolve_existing_path(path, bases):
    if path is None:
        return None
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return None


def load_python_model_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    variables = runpy.run_path(str(config_path))
    model_config = variables.get('model')
    if not isinstance(model_config, dict):
        raise ValueError(f'Config has no model dictionary: {config_path}')
    return dict(model_config), config_path


def load_model_config(args, best_params_dir):
    run_root, snapshot_path, snapshot = find_run_snapshot(best_params_dir)
    if args.config:
        model_config, config_source = load_python_model_config(args.config)
    elif snapshot is not None:
        model_config = snapshot.get('config', {}).get('model')
        if not isinstance(model_config, dict):
            raise ValueError(
                f'Run snapshot has no config.model object: {snapshot_path}')
        model_config = dict(model_config)
        config_source = snapshot_path
    else:
        raise FileNotFoundError(
            'Could not find config_snapshot.yaml near prompt_best_params. '
            'Pass --config explicitly.')

    if args.concept_path:
        concept_path = Path(args.concept_path).expanduser().resolve()
        if not concept_path.is_file():
            raise FileNotFoundError(f'Concept JSON not found: {concept_path}')
    else:
        concept_path = None
        if snapshot is not None:
            original = snapshot.get('concept_json_path')
            if original:
                copied = run_root / 'config_snapshot' / Path(original).name
                if copied.is_file():
                    concept_path = copied.resolve()
                else:
                    concept_path = resolve_existing_path(
                        original, [run_root, REPO_ROOT])
        if concept_path is None:
            concept_path = resolve_existing_path(
                model_config.get('classname_path'),
                [Path(config_source).parent, REPO_ROOT])
        if concept_path is None:
            raise FileNotFoundError(
                'Could not resolve model.classname_path. Pass --concept-path.')

    model_config['classname_path'] = str(concept_path)
    model_config['device'] = torch.device(args.device)
    model_config['freeze_sam'] = True
    model_config.pop('type', None)
    model_config.pop('model_type', None)
    return model_config, config_source, concept_path, snapshot_path


def validate_export_configuration(model_config):
    fusion_mode = model_config.get('fusion_mode', 'max')
    if fusion_mode not in ('weighted', 'weighted_sum'):
        raise ValueError(
            'The Stage A final_score formula requires fusion_mode='
            f'weighted_sum, but the restored value is {fusion_mode!r}.')
    parameterization = model_config.get(
        'prompt_parameterization', 'four_parameter')
    if parameterization != 'identifiable_three':
        raise ValueError(
            'This exporter requires prompt_parameterization='
            f'identifiable_three, but found {parameterization!r}.')
    if not model_config.get('use_presence_score', True):
        raise ValueError(
            'The Stage A final_score formula requires use_presence_score=True.')


def configure_cuda(device_name):
    device = torch.device(device_name)
    if device.type != 'cuda':
        raise ValueError(
            'SAM3 construction and head extraction currently require CUDA; '
            f'got --device {device_name!r}.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available, so SAM3 cannot be exported.')
    if device.index is not None:
        torch.cuda.set_device(device.index)
    return device


def build_model(model_config):
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        model = SegEarthOV3LearnableSegmentation(**model_config)
    finally:
        os.chdir(previous_cwd)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def scalar_parameter(parameters, name):
    if name not in parameters:
        raise ValueError(f'Selected parameters miss {name!r}.')
    value = parameters[name]
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f'Parameter {name!r} is not scalar: {value.shape}')
        value = value.detach().cpu().item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'Parameter {name!r} is not finite: {value}')
    return value


def validate_and_apply_checkpoints(model, selected):
    prompt_keys = model._prompt_keys()
    for item in selected:
        index = item.prompt_index
        if index < 0 or index >= model.num_queries:
            raise ValueError(
                f'Checkpoint prompt index {index} is outside model range.')
        model_prompt, model_concept, _ = prompt_keys[index]
        if model_prompt != item.prompt or model_concept != item.concept_id:
            raise ValueError(
                f'Checkpoint/model prompt mismatch at index {index}: '
                f'checkpoint=({item.prompt!r}, {item.concept_id!r}), '
                f'model=({model_prompt!r}, {model_concept!r}).')
        score_version = item.checkpoint.get(
            'prompt_score_version', item.record.get('prompt_score_version'))
        if score_version != model.PROMPT_SCORE_VERSION:
            raise ValueError(
                f'Prompt {item.prompt!r} uses score version '
                f'{score_version!r}, model uses {model.PROMPT_SCORE_VERSION!r}.')
        parameterization = item.record.get('prompt_parameterization')
        if parameterization and parameterization != model.prompt_parameterization:
            raise ValueError(
                f'Prompt {item.prompt!r} uses parameterization '
                f'{parameterization!r}, model uses '
                f'{model.prompt_parameterization!r}.')

        semantic_weight = scalar_parameter(item.parameters, 'semantic_weight')
        presence_gain = scalar_parameter(item.parameters, 'presence_gain')
        presence_bias = scalar_parameter(item.parameters, 'presence_bias')
        prompt_weight = scalar_parameter(item.parameters, 'prompt_weight')
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError(
                f'Invalid semantic_weight for {item.prompt!r}: '
                f'{semantic_weight}')
        if presence_gain < 0.0:
            raise ValueError(
                f'Invalid presence_gain for {item.prompt!r}: {presence_gain}')
        if not 0.0 <= presence_bias <= 1.0:
            raise ValueError(
                f'Invalid presence_bias for {item.prompt!r}: {presence_bias}')
        if prompt_weight <= 0.0:
            raise ValueError(
                f'Invalid prompt_weight for {item.prompt!r}: {prompt_weight}')

        model.set_prompt_parameter_group(
            index, item.optimization_parameters)
        bounded = model._bounded_prompt_values()
        actual = {
            'semantic_weight': bounded['semantic_weight'][index].item(),
            'presence_gain': bounded['presence_gain'][index].item(),
            'presence_bias': bounded['presence_bias'][index].item(),
            'prompt_weight': bounded['prompt_weight'][index].item(),
        }
        expected = {
            'semantic_weight': semantic_weight,
            'presence_gain': presence_gain,
            'presence_bias': presence_bias,
            'prompt_weight': prompt_weight,
        }
        for name in expected:
            if not math.isclose(
                    actual[name], expected[name], rel_tol=2e-6, abs_tol=2e-6):
                raise ValueError(
                    f'Applied {name} mismatch for {item.prompt!r}: '
                    f'checkpoint={expected[name]}, model={actual[name]}')


def reconstruct_final_score(heads, parameters):
    """Apply the exact weighted Stage A final_score reconstruction rule."""
    instance_score = heads['instance_score_map'].float()
    semantic_score = heads['semantic_score_map'].float()
    if instance_score.shape != semantic_score.shape:
        raise ValueError(
            'Prompt head shape mismatch: '
            f'instance={tuple(instance_score.shape)}, '
            f'semantic={tuple(semantic_score.shape)}')
    semantic_weight = scalar_parameter(parameters, 'semantic_weight')
    presence_gain = scalar_parameter(parameters, 'presence_gain')
    presence_bias = scalar_parameter(parameters, 'presence_bias')
    prompt_weight = scalar_parameter(parameters, 'prompt_weight')

    fused = (
        semantic_weight * semantic_score
        + (1.0 - semantic_weight) * instance_score)
    fused = fused.clamp(SCORE_EPS, 1.0 - SCORE_EPS)
    final_logit = (
        torch.logit(fused)
        + presence_gain * (heads['presence_score'].float() - presence_bias)
        + math.log(prompt_weight))
    final_score = torch.sigmoid(final_logit)
    if not torch.isfinite(final_score).all():
        raise ValueError('Reconstructed final_score contains non-finite values.')
    if torch.any(final_score < 0.0) or torch.any(final_score > 1.0):
        raise ValueError('Reconstructed final_score is outside [0, 1].')
    return final_score


def panel_map(value):
    return (
        value.detach().float().squeeze().clamp(0.0, 1.0).cpu().numpy())


def visual_output_path(vis_folder, image_path, item):
    image_id = safe_prompt_folder(image_path.stem)
    prompt_name = safe_prompt_folder(item.prompt)
    return vis_folder / (
        f'{image_id}_[{prompt_name}]_[epoch_{item.best_epoch:04d}].png')


def save_prompt_visualization(save_panel, model, image, heads, final_score,
                              item, output_path):
    """Render the same no-GT diagnostic panel used by prompt training."""
    instance_score = heads['instance_score_map'].float()
    semantic_score = heads['semantic_score_map'].float()
    presence_score = heads['presence_score'].detach().float().squeeze()
    semantic_weight = scalar_parameter(
        item.parameters, 'semantic_weight')
    presence_gain = scalar_parameter(item.parameters, 'presence_gain')
    presence_bias = scalar_parameter(item.parameters, 'presence_bias')

    fused = (
        semantic_weight * semantic_score
        + (1.0 - semantic_weight) * instance_score)
    presence_logit = (
        presence_gain * (presence_score - presence_bias)).clamp(-8.0, 8.0)
    instance_gated = torch.sigmoid(
        model.probability_to_logits(instance_score) + presence_logit)

    save_panel(
        image=image,
        semantic=panel_map(semantic_score),
        instance_raw=panel_map(instance_score),
        instance_gated=panel_map(instance_gated),
        fused_before_presence=panel_map(fused),
        final_score=panel_map(final_score),
        presence_score=float(presence_score.cpu()),
        prompt=f'{item.prompt} | epoch {item.best_epoch}',
        output_path=output_path)


def collect_images(data_dir, extensions, recursive, max_images):
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Input data directory not found: {data_dir}')
    normalized_extensions = {
        extension.lower()
        if extension.startswith('.') else f'.{extension.lower()}'
        for extension in extensions
    }
    if not normalized_extensions:
        raise ValueError('--extensions must contain at least one extension.')
    iterator = data_dir.rglob('*') if recursive else data_dir.iterdir()
    images = sorted(
        (path for path in iterator
         if path.is_file() and path.suffix.lower() in normalized_extensions),
        key=lambda path: path.relative_to(data_dir).as_posix().casefold())
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(
            f'No matching images found in {data_dir}; '
            f'extensions={sorted(normalized_extensions)}')
    stems = {}
    for path in images:
        key = path.stem.casefold()
        if key in stems:
            raise ValueError(
                'Flat prompt output cannot represent duplicate image stems: '
                f'{stems[key]} and {path}')
        stems[key] = path
    return data_dir, images


def score_output_path(output_dir, image_path, item):
    return (
        output_dir / safe_prompt_folder(item.prompt)
        / f'{image_path.stem}.npy')


def preflight_outputs(output_dir, selected):
    folders = {}
    for item in selected:
        folder_name = safe_prompt_folder(item.prompt)
        key = folder_name.casefold()
        if key in folders:
            raise ValueError(
                'Selected prompts map to the same required output folder '
                f'{folder_name!r}: indices {folders[key]} and '
                f'{item.prompt_index}. Export them in separate runs.')
        folders[key] = item.prompt_index
    return output_dir / 'export_manifest.json'


def load_existing_score(path, expected_shape):
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(
            f'Cannot read existing score {path}; use --overwrite to rebuild '
            f'it: {error}') from error
    if values.shape != expected_shape:
        raise ValueError(
            f'Existing score shape mismatch for {path}: {values.shape} != '
            f'{expected_shape}. Use --overwrite to rebuild it.')
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(
            f'Existing score contains non-finite values: {path}. Use '
            '--overwrite to rebuild it.')
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(
            f'Existing score is outside [0, 1]: {path}. Use --overwrite '
            'to rebuild it.')
    return values


def atomic_save_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    try:
        with temporary.open('wb') as file:
            np.save(file, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_safe(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as file:
            json.dump(json_safe(manifest), file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_scores(args):
    best_params_dir = resolve_best_params_dir(args.best_params_dir)
    checkpoints = load_prompt_checkpoints(best_params_dir)
    selected = resolve_prompt_selection(args.prompts, checkpoints)
    model_config, config_source, concept_path, snapshot_path = (
        load_model_config(args, best_params_dir))
    validate_export_configuration(model_config)

    data_dir, images = collect_images(
        args.data_dir, args.extensions, args.recursive, args.max_images)
    output_dir = Path(args.output_dir).expanduser().resolve()
    vis_folder = (
        Path(args.vis_folder).expanduser().resolve()
        if args.vis_folder else None)
    manifest_path = preflight_outputs(output_dir, selected)
    pending_score_count = sum(
        args.overwrite
        or not score_output_path(output_dir, image_path, item).exists()
        for image_path in images for item in selected)
    pending_visual_count = (
        sum(
            args.overwrite
            or not visual_output_path(vis_folder, image_path, item).exists()
            for image_path in images for item in selected)
        if vis_folder is not None else 0)
    work_required = pending_score_count > 0 or pending_visual_count > 0
    device = (
        configure_cuda(args.device)
        if work_required else torch.device(args.device))

    print(f'best_params_dir={best_params_dir}', flush=True)
    print(f'config_source={config_source}', flush=True)
    print(f'concept_path={concept_path}', flush=True)
    print(f'data_dir={data_dir}', flush=True)
    print(f'output_dir={output_dir}', flush=True)
    print(f'vis_folder={vis_folder}', flush=True)
    print(
        'prompts=' + ', '.join(
            f'{item.prompt!r}@{item.concept_id}[{item.prompt_index}]'
            for item in selected),
        flush=True)
    print(
        f'images={len(images)} dtype={args.dtype} device={device}',
        flush=True)
    print(
        f'pending_scores={pending_score_count} '
        f'pending_panels={pending_visual_count} overwrite={args.overwrite}',
        flush=True)

    if work_required:
        model = build_model(model_config)
        validate_and_apply_checkpoints(model, selected)
    else:
        model = None
        print(
            'All requested score files and panels already exist; '
            'SAM3 inference will be skipped.',
            flush=True)
    if pending_visual_count > 0:
        from diagnose_sam3_heads import save_panel
        vis_folder.mkdir(parents=True, exist_ok=True)
    else:
        save_panel = None
    output_dtype = np.float16 if args.dtype == 'float16' else np.float32
    channel_stats = {item.prompt_index: RunningStats() for item in selected}
    maximum_score_difference = {item.prompt_index: 0.0 for item in selected}
    verified_image_count = {item.prompt_index: 0 for item in selected}
    generated_score_count = {item.prompt_index: 0 for item in selected}
    skipped_score_count = {item.prompt_index: 0 for item in selected}
    generated_visual_count = {item.prompt_index: 0 for item in selected}
    skipped_visual_count = {item.prompt_index: 0 for item in selected}
    image_records = []

    with torch.inference_mode():
        for image_number, image_path in enumerate(images, start=1):
            item_plan = {}
            pending_items = []
            for item in selected:
                output_path = score_output_path(
                    output_dir, image_path, item)
                panel_path = (
                    visual_output_path(vis_folder, image_path, item)
                    if vis_folder is not None else None)
                needs_score = args.overwrite or not output_path.exists()
                needs_visual = (
                    panel_path is not None
                    and (args.overwrite or not panel_path.exists()))
                item_plan[item.prompt_index] = {
                    'output_path': output_path,
                    'panel_path': panel_path,
                    'needs_score': needs_score,
                    'needs_visual': needs_visual,
                }
                if needs_score or needs_visual:
                    pending_items.append(item)

            with Image.open(image_path) as source:
                width, height = source.size
                image = source.convert('RGB') if pending_items else None
            expected_shape = (height, width)

            computed = {}
            if pending_items:
                query_indices = [
                    item.prompt_index for item in pending_items]
                heads_batch = model.extract_prompt_heads_batch(
                    image, query_indices, ori_shape=expected_shape)
                if len(heads_batch) != len(pending_items):
                    raise RuntimeError(
                        f'Expected {len(pending_items)} prompt heads, got '
                        f'{len(heads_batch)} for {image_path}.')
                for item, heads in zip(pending_items, heads_batch):
                    final_score = reconstruct_final_score(
                        heads, item.parameters)
                    verification_heads = {
                        name: value.float() if torch.is_tensor(value) else value
                        for name, value in heads.items()
                    }
                    model_score = model.calibrate_prompt_heads(
                        item.prompt_index, verification_heads).float()
                    difference = float(torch.max(torch.abs(
                        final_score - model_score)).item())
                    maximum_score_difference[item.prompt_index] = max(
                        maximum_score_difference[item.prompt_index],
                        difference)
                    verified_image_count[item.prompt_index] += 1
                    if difference > args.verify_atol:
                        raise RuntimeError(
                            f'Final-score verification failed for image '
                            f'{image_path.name}, prompt {item.prompt!r}: '
                            f'max_abs_diff={difference:.9g} > '
                            f'atol={args.verify_atol:.9g}')
                    values = final_score.detach().cpu().numpy().astype(
                        np.float32, copy=False)
                    if values.shape != expected_shape:
                        raise RuntimeError(
                            f'Output shape mismatch for {image_path.name}, '
                            f'{item.prompt!r}: {values.shape} != '
                            f'{expected_shape}')
                    computed[item.prompt_index] = {
                        'final_score': final_score,
                        'heads': verification_heads,
                        'values': values,
                    }

            outputs = []
            generated_outputs = []
            skipped_outputs = []
            visualizations = []
            generated_visualizations = []
            skipped_visualizations = []
            for item in selected:
                plan = item_plan[item.prompt_index]
                output_path = plan['output_path']
                relative_output = str(output_path.relative_to(output_dir))
                outputs.append(relative_output)
                if plan['needs_score']:
                    values = computed[item.prompt_index]['values']
                    atomic_save_npy(
                        output_path, values.astype(output_dtype, copy=False))
                    generated_outputs.append(relative_output)
                    generated_score_count[item.prompt_index] += 1
                else:
                    values = load_existing_score(output_path, expected_shape)
                    skipped_outputs.append(relative_output)
                    skipped_score_count[item.prompt_index] += 1
                channel_stats[item.prompt_index].update(values)

                if vis_folder is not None:
                    panel_path = plan['panel_path']
                    visualizations.append(str(panel_path))
                    if plan['needs_visual']:
                        current = computed[item.prompt_index]
                        save_prompt_visualization(
                            save_panel,
                            model,
                            image,
                            current['heads'],
                            current['final_score'],
                            item,
                            panel_path)
                        generated_visualizations.append(str(panel_path))
                        generated_visual_count[item.prompt_index] += 1
                    else:
                        skipped_visualizations.append(str(panel_path))
                        skipped_visual_count[item.prompt_index] += 1

            image_records.append({
                'source': str(image_path),
                'image_name': image_path.name,
                'output_name': f'{image_path.stem}.npy',
                'width': width,
                'height': height,
                'outputs': outputs,
                'generated_outputs': generated_outputs,
                'skipped_outputs': skipped_outputs,
                'visualizations': visualizations,
                'generated_visualizations': generated_visualizations,
                'skipped_visualizations': skipped_visualizations,
            })
            if (image_number == 1
                    or image_number % args.progress_interval == 0
                    or image_number == len(images)):
                print(
                    f'[{image_number}/{len(images)}] {image_path.name} '
                    f'scores=new:{len(generated_outputs)} '
                    f'skipped:{len(skipped_outputs)} '
                    f'panels=new:{len(generated_visualizations)} '
                    f'skipped:{len(skipped_visualizations)}',
                    flush=True)

    channels = []
    for channel_index, item in enumerate(selected):
        channels.append({
            'channel_index': channel_index,
            'prompt_index': item.prompt_index,
            'prompt': item.prompt,
            'concept_id': item.concept_id,
            'concept_name': item.concept_name,
            'output_folder': safe_prompt_folder(item.prompt),
            'checkpoint': str(item.path),
            'best_epoch': item.best_epoch,
            'selected_epoch': item.record.get(
                'selected_epoch', item.record.get('best_epoch')),
            'prompt_exit': item.record.get('prompt_exit'),
            'parameters': item.parameters,
            'optimization_parameters': item.optimization_parameters,
            'statistics_float32': channel_stats[item.prompt_index].as_dict(),
            'max_abs_diff_formula_vs_model': (
                maximum_score_difference[item.prompt_index]
                if verified_image_count[item.prompt_index] > 0 else None),
            'verified_image_count': verified_image_count[item.prompt_index],
            'generated_score_count': generated_score_count[item.prompt_index],
            'skipped_score_count': skipped_score_count[item.prompt_index],
            'generated_visual_count': (
                generated_visual_count[item.prompt_index]),
            'skipped_visual_count': skipped_visual_count[item.prompt_index],
        })

    manifest = {
        'format_version': 1,
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'vis_folder': str(vis_folder) if vis_folder is not None else None,
        'best_params_dir': str(best_params_dir),
        'config_source': str(config_source),
        'config_snapshot': (
            str(snapshot_path) if snapshot_path is not None else None),
        'concept_path': str(concept_path),
        'device': str(device),
        'dtype': args.dtype,
        'array_layout': 'H,W',
        'value_range': [0.0, 1.0],
        'continuous_scores': True,
        'threshold_applied': False,
        'score_epsilon': SCORE_EPS,
        'score_formula': FINAL_SCORE_FORMULA,
        'formula_verification_atol': args.verify_atol,
        'image_count': len(image_records),
        'channel_count': len(channels),
        'channels': channels,
        'images': image_records,
        'arguments': vars(args),
    }
    write_manifest(manifest_path, manifest)
    print(f'manifest={manifest_path}', flush=True)
    for channel in channels:
        stats = channel['statistics_float32']
        difference = channel['max_abs_diff_formula_vs_model']
        difference_text = 'n/a' if difference is None else f'{difference:.3g}'
        print(
            f"[channel] {channel['prompt']!r} "
            f"min={stats['min']:.6f} max={stats['max']:.6f} "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"scores=new:{channel['generated_score_count']} "
            f"skipped:{channel['skipped_score_count']} "
            f"panels=new:{channel['generated_visual_count']} "
            f"skipped:{channel['skipped_visual_count']} "
            f"verify_diff={difference_text}",
            flush=True)
    print('Prompt final_score export completed.', flush=True)


def main():
    args = parse_args()
    try:
        export_scores(args)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    main()
