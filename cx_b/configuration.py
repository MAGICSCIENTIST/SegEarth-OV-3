"""Configuration registry and validation for CX-B method combinations."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os.path as osp
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_METHODS = ('A3-2', 'dynamic_head', 'segov3')
PROMPT_INIT_MODES = ('scratch', 'a3_warmstart', 'a3_reuse')
CONCEPT_METHODS = (
    'A3-2', 'max', 'logsumexp', 'dynamic', 'reliability_dynamic',
    'region_dynamic', 'sam_region_dynamic')
LOSS_METHODS = ('A3-2', 'soft_iou_bce', 'bce_tversky', 'ce_lovasz', 'lovasz')
JOINT_MODES = ('independent', 'concept_prompt', 'multiclass_concept',
               'multiclass_prompt', 'multiclass_joint')


def refiner_settings(cx):
    method = cx.get('joint_refiner', 'none')
    strength = cx.get('joint_refiner_strength', 1.0)
    if method not in ('none', 'ov_class_spatial'):
        raise ValueError('joint_refiner must be none or ov_class_spatial')
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('joint_refiner_strength must be in [0,1]')
    return ('none', 0.) if method == 'none' or strength == 0 else (method, float(strength))


def effective_joint_train(cx):
    config = dict(cx['joint_train'])
    if refiner_settings(cx)[0] != 'none':
        config.update(epochs=5, patience=3, lr=1e-4, loss_method='ce_lovasz',
                      lovasz_weight=.25, binary_weight=0.)
        if cx.get('loss_method') == 'lovasz':
            config.update(loss_method='lovasz', competition_weight=0.,
                          binary_weight=0., lovasz_weight=1.)
        config.update(cx.get('refiner_train', {}))
        if config.get('binary_weight') != 0:
            raise ValueError('refiner-only training requires binary_weight=0')
        for key in ('threshold', 'background', 'temperature'):
            if config[key] != cx['joint_train'][key]:
                raise ValueError('refiner_train cannot change the base decision: '+key)
    return config


def validate_pure_lovasz(config):
    """Do not allow a run named lovasz to silently optimize a mixed objective."""
    if config.get('loss_method') != 'lovasz':
        raise ConfigurationError('lovasz requires loss_method=lovasz in the training settings')
    for key, expected in (('competition_weight', 0.), ('binary_weight', 0.),
                          ('lovasz_weight', 1.)):
        if config.get(key) != expected:
            raise ConfigurationError(f'Pure lovasz requires {key}={expected}; use ce_lovasz for a mixed loss')


def slug(value):
    return str(value).lower().replace('-', '_')


def combination_name(prompt_method, concept_method, loss_method):
    return (f'p_{slug(prompt_method)}__c_{slug(concept_method)}__'
            f'l_{slug(loss_method)}')


COMBINATION_CONFIGS = {
    (prompt, concept, loss): str(
        (REPO_ROOT / 'configs' / 'cx_b' /
         f'{combination_name(prompt, concept, loss)}.py').resolve())
    for prompt in PROMPT_METHODS
    for concept in CONCEPT_METHODS
    for loss in LOSS_METHODS
}


class ConfigurationError(ValueError):
    pass


def registered_config(prompt_method, concept_method, loss_method):
    key = (prompt_method, concept_method, loss_method)
    if key not in COMBINATION_CONFIGS:
        raise ConfigurationError(
            f'Unregistered CX-B combination {key}. Use --list-methods.')
    path = COMBINATION_CONFIGS[key]
    if not osp.isfile(path):
        raise ConfigurationError(f'Registered config does not exist: {path}')
    return path


def parse_cfg_options(items):
    overrides = {}
    for item in items or []:
        if '=' not in item:
            raise ConfigurationError(f'Invalid --cfg-options item: {item!r}')
        key, raw = item.split('=', 1)
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        overrides[key] = value
    return overrides


PROTECTED_OVERRIDE_KEYS = {
    'cx_b.prompt_method', 'cx_b.prompt_methd', 'cx_b.concept_method',
    'cx_b.loss_method', 'cx_b.data_mode', 'cx_b.concept_independent',
    'cx_b.sam_region_cfg.read_output_cache',
    'cx_b.sam_region_cfg.write_output_cache',
}


def apply_dotted_overrides(config, overrides):
    for dotted_key, value in overrides.items():
        if dotted_key in PROTECTED_OVERRIDE_KEYS:
            raise ConfigurationError(
                f'--cfg-options cannot override CX-B identity/contract: '
                f'{dotted_key}')
        target = config
        parts = dotted_key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return config


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (key in result and isinstance(result[key], dict)
                and isinstance(value, dict)):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_python_config(path):
    """Load the small Python configs with deterministic _base_ inheritance."""
    path = Path(path).resolve()
    namespace = runpy.run_path(str(path))
    own = {key: value for key, value in namespace.items()
           if not key.startswith('_') and not callable(value)}
    base_value = namespace.get('_base_')
    if base_value is None:
        return own
    bases = [base_value] if isinstance(base_value, str) else list(base_value)
    merged = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = deep_merge(merged, load_python_config(base_path))
    return deep_merge(merged, own)


def resolve_combination_config(prompt_method, concept_method, loss_method,
                               overrides=None, alternate_base=None):
    path = registered_config(prompt_method, concept_method, loss_method)
    config = load_python_config(path)
    if alternate_base:
        # An alternate dataset/base config supplies project data/model fields;
        # CX-B method identity remains owned by the registered combination.
        alternate = load_python_config(alternate_base)
        cx = config['cx_b']
        config = deep_merge(config, alternate)
        config['cx_b'] = cx
    apply_dotted_overrides(config, overrides or {})
    validate_config(config, (prompt_method, concept_method, loss_method))
    return config, path


def validate_config(config, expected=None):
    cx = config.get('cx_b')
    if not isinstance(cx, dict):
        raise ConfigurationError('Config must define a cx_b dictionary.')
    identity = (cx.get('prompt_method'), cx.get('concept_method'),
                cx.get('loss_method'))
    if expected is not None and tuple(expected) != identity:
        raise ConfigurationError(
            f'Config identity {identity} does not match CLI {tuple(expected)}.')
    if identity[0] not in PROMPT_METHODS:
        raise ConfigurationError(f'Invalid prompt method: {identity[0]}')
    if identity[1] not in CONCEPT_METHODS:
        raise ConfigurationError(f'Invalid concept method: {identity[1]}')
    if identity[2] not in LOSS_METHODS:
        raise ConfigurationError(f'Invalid loss method: {identity[2]}')
    prompt_init = cx.get('prompt_init', 'scratch')
    if prompt_init not in PROMPT_INIT_MODES:
        raise ConfigurationError(
            f'Invalid prompt_init: {prompt_init!r}. Expected one of '
            f'{PROMPT_INIT_MODES}.')
    if identity[0] == 'segov3' and prompt_init != 'scratch':
        raise ConfigurationError(
            'segov3 is a fixed scoring rule; use prompt_init=scratch, '
            'without an A3 warmstart/reuse checkpoint.')
    if cx.get('cache_dtype', 'float16') not in ('float16', 'float32'):
        raise ConfigurationError('cx_b.cache_dtype must be float16 or float32.')
    if cx.get('concept_independent') is not True:
        raise ConfigurationError('CX-B requires concept_independent=True.')
    joint_mode = cx.get('joint_mode', 'independent')
    if joint_mode not in JOINT_MODES:
        raise ConfigurationError(f'Invalid joint_mode: {joint_mode!r}')
    try:
        refiner, _ = refiner_settings(cx)
        if refiner != 'none':
            if not joint_mode.startswith('multiclass_'):
                raise ValueError('refiner requires a multiclass joint mode')
            train = effective_joint_train(cx)
            for field in ('epochs', 'patience'):
                if isinstance(train[field], bool) or not isinstance(train[field], int) or train[field] < 0:
                    raise ValueError('Invalid refiner_train.'+field)
            if not math.isfinite(train['lr']) or train['lr'] <= 0:
                raise ValueError('Invalid refiner_train.lr')
            if train['loss_method'] not in ('ce', 'ce_lovasz', 'lovasz') or not math.isfinite(train['lovasz_weight']) or train['lovasz_weight'] < 0:
                raise ValueError('Invalid refiner loss configuration')
            if identity[2] == 'lovasz' or train['loss_method'] == 'lovasz':
                validate_pure_lovasz(train)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    if identity[2] in ('ce_lovasz', 'lovasz'):
        joint = cx.get('joint_train', {})
        if joint_mode not in ('multiclass_concept', 'multiclass_prompt', 'multiclass_joint'):
            raise ConfigurationError(f'{identity[2]} requires a multiclass joint mode')
        if joint.get('loss_method') != identity[2]:
            raise ConfigurationError(f'{identity[2]} requires joint_train.loss_method={identity[2]}')
        weight = joint.get('lovasz_weight', .25)
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ConfigurationError('joint_train.lovasz_weight must be finite and non-negative')
        if identity[2] == 'lovasz':
            validate_pure_lovasz(joint)
    elif cx.get('joint_train', {}).get('loss_method', 'ce') != 'ce':
        raise ConfigurationError('Select --loss_method=ce_lovasz or lovasz to enable joint Lovasz')
    if joint_mode != 'independent':
        joint = cx.get('joint_train', {})
        for key in ('epochs', 'patience'):
            value = joint.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(f'joint_train.{key} must be a non-negative integer')
        if joint.get('lr', 0) <= 0:
            raise ConfigurationError('joint_train.lr must be positive')
        if not 0 < joint.get('threshold', .5) < 1:
            raise ConfigurationError('joint_train.threshold must be in (0,1)')
        if joint.get('temperature', 1.) <= 0:
            raise ConfigurationError('joint_train.temperature must be positive')
        if identity[2] != 'lovasz' and (joint.get('binary_weight', .25) < 0 or joint.get('competition_weight', 1.) <= 0):
            raise ConfigurationError('Joint binary weight must be >=0 and competition weight >0')
        if (isinstance(joint.get('background', 0), bool)
                or not isinstance(joint.get('background', 0), int)
                or joint.get('background', 0) < 0):
            raise ConfigurationError('joint_train.background must be a non-negative class index')
    if not isinstance(cx.get('export_probabilities', False), bool):
        raise ConfigurationError('cx_b.export_probabilities must be bool.')
    for section_name in ('prompt_train', 'concept_train', 'joint_train'):
        train_cfg = cx.get(section_name, {})
        checkpoint_interval = train_cfg.get(
            'checkpoint_interval_epochs', 0)
        if (isinstance(checkpoint_interval, bool)
                or not isinstance(checkpoint_interval, int)
                or checkpoint_interval < 0):
            raise ConfigurationError(
                f'cx_b.{section_name}.checkpoint_interval_epochs must be '
                'a non-negative integer.')
        if not isinstance(train_cfg.get('save_last_checkpoint', True), bool):
            raise ConfigurationError(
                f'cx_b.{section_name}.save_last_checkpoint must be bool.')
        validate_every = train_cfg.get('validate_every_epochs', 1)
        if (isinstance(validate_every, bool)
                or not isinstance(validate_every, int)
                or validate_every <= 0):
            raise ConfigurationError(
                f'cx_b.{section_name}.validate_every_epochs must be a '
                'positive integer.')
    expected_mode = ('online_sam' if identity[1] == 'sam_region_dynamic'
                     else 'cache')
    if cx.get('data_mode') != expected_mode:
        raise ConfigurationError(
            f'{identity[1]} requires data_mode={expected_mode!r}.')
    region_cfg = cx.get('sam_region_cfg', {})
    if region_cfg.get('read_output_cache') or region_cfg.get(
            'write_output_cache'):
        raise ConfigurationError(
            'sam_region_dynamic forbids region/head output caches.')
    if region_cfg.get('amp_dtype', 'bfloat16') not in (
            'bfloat16', 'float16', 'float32'):
        raise ConfigurationError('Invalid sam_region_cfg.amp_dtype.')
    if region_cfg.get('quality_source', 'object_score') not in (
            'object_score', 'object_score_raw'):
        raise ConfigurationError('Invalid sam_region_cfg.quality_source.')
    return config


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), default=str)


def config_fingerprint(config, selected_inputs=None):
    payload = dict(config=copy.deepcopy(config), inputs=selected_inputs or {})
    return hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()
