"""Validate a complete joint source before starting a new fine-tuning run."""
from __future__ import annotations

import json
import os.path as osp
from pathlib import Path

import torch

from .checkpoints import sha256_file, prompt_identity
from .modules import load_variant
from .configuration import refiner_settings
from .refiners import JOINT_FORMATS, validate_joint_refiner


class InitializedModules(list):
    refiner_payload = None
    base_metrics = None


def load_joint_initialization(path, config, records, cache_root, slots):
    """Return CPU modules and provenance, without touching the source run.

    slots is the ordered list of target-relative prompt then concept paths.
    A source snapshot is required to check cache/model semantics, not just shapes.
    """
    path = Path(path).resolve()
    snapshot_path = path.parent.parent/'run_config.json'
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    bundle = torch.load(path, map_location='cpu')
    cx, old = config['cx_b'], snapshot['config']
    digest = sha256_file(path)
    if cx.get('joint_init_sha256', digest) != digest:
        raise ValueError('Joint initialization checkpoint changed after CLI resolution')
    if (bundle.get('format') not in JOINT_FORMATS or not bundle.get('run_fingerprint')
            or bundle['run_fingerprint'] != snapshot.get('fingerprint')):
        raise ValueError('Joint initialization checkpoint format/run mismatch')
    validate_joint_refiner(bundle)
    for key in ('prompt_method', 'concept_method', 'data_mode'):
        if cx[key] != old['cx_b'][key]:
            raise ValueError(f'Joint initialization {key} mismatch')
    for key in ('confidence_threshold', 'use_sem_seg', 'use_transformer_decoder',
                'instance_filter_use_presence'):
        if config.get('model', {}).get(key) != old.get('model', {}).get(key):
            raise ValueError(f'Joint initialization head setting mismatch: {key}')
    if cx['data_mode'] == 'cache':
        old_cache = snapshot.get('inputs', {}).get('prompt_head_cache')
        if not old_cache or osp.normcase(osp.abspath(old_cache)) != osp.normcase(osp.abspath(cache_root)):
            raise ValueError('Joint initialization prompt head cache mismatch')
    for key in ('threshold', 'background'):
        if bundle.get('decision', {}).get(key) != cx['joint_train'][key]:
            raise ValueError(f'Joint initialization decision mismatch: {key}')
    modules = bundle.get('modules', {})
    if set(slots) != set(modules):
        raise ValueError('Joint initialization missing/extra prompt or concept modules')
    loaded = InitializedModules()
    refiner_method, strength = refiner_settings(cx)
    if bundle['format'] == 'cx_b_joint_v2':
        if refiner_method != 'none':
            if strength != bundle['refiner']['strength']:
                raise ValueError('Resume refiner with its saved strength; use evaluation overrides for ablation')
            loaded.refiner_payload = bundle['refiner']
        loaded.base_metrics = bundle['base_metrics']
    for index, slot in enumerate(slots):
        payload = modules[slot]
        meta = payload.get('metadata', {})
        kind = 'prompt' if index < len(records) else 'concept'
        if (meta.get('kind') != kind or meta.get('method') != cx[kind+'_method']
                or meta.get('run_fingerprint') != bundle['run_fingerprint']
                or meta.get('joint_epoch') != bundle.get('epoch')
                or meta.get('joint_mode') != bundle.get('mode')):
            raise ValueError(f'Joint initialization mixed method/run/epoch: {slot}')
        if kind == 'prompt':
            saved, expected = [meta.get('identity', {})], [records[index]]
        else:
            saved = meta.get('members', [])
            expected = [r for r in records if r['concept_index'] == index-len(records)]
            if meta.get('prompt_count') != len(expected):
                raise ValueError(f'Joint initialization prompt count mismatch: {slot}')
        if (len(saved) != len(expected) or any(
                prompt_identity(a) != prompt_identity(b)
                or a.get('prompt_index') != b['prompt_index']
                or a.get('concept_index') != b['concept_index']
                for a, b in zip(saved, expected))):
            raise ValueError(f'Joint initialization identity/membership mismatch: {slot}')
        method = cx[kind+'_method']
        if (method in ('dynamic_head', 'dynamic', 'reliability_dynamic', 'region_dynamic', 'sam_region_dynamic')
                and meta.get('hidden_channels', 16) != cx[kind+'_train']['hidden_channels']):
            raise ValueError(f'Joint initialization hidden_channels mismatch: {slot}')
        loaded.append(load_variant(slot, payload=payload))
    provenance = dict(path=str(path), sha256=digest, snapshot=str(snapshot_path),
                      snapshot_sha256=sha256_file(snapshot_path),
                      source_fingerprint=bundle['run_fingerprint'], source_epoch=bundle['epoch'],
                      source_mode=bundle['mode'], source_metrics=bundle.get('metrics'),
                      optimizer='new', skipped_stages=['prompt', 'concept'])
    if bundle['format'] == 'cx_b_joint_v2' and refiner_method == 'none':
        provenance.update(source_metrics=bundle['base_metrics'], refiner_disabled=True)
    return loaded, provenance
