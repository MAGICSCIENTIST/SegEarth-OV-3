"""Checkpoint identity, fingerprints, and atomic persistence for CX-B."""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import tempfile
from pathlib import Path

import torch


PROMPT_SCORE_VERSION = 'identifiable_three_v1'


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value, path):
    path = osp.abspath(path)
    os.makedirs(osp.dirname(path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=osp.basename(path) + '.', suffix='.tmp', dir=osp.dirname(path))
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if osp.exists(temporary):
            os.unlink(temporary)


def atomic_json_save(value, path):
    path = osp.abspath(path)
    os.makedirs(osp.dirname(path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=osp.basename(path) + '.', suffix='.tmp', dir=osp.dirname(path),
        text=True)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, default=str)
        os.replace(temporary, path)
    finally:
        if osp.exists(temporary):
            os.unlink(temporary)


def prompt_identity(record):
    return (
        str(record.get('prompt')),
        str(record.get('concept_id')),
        int(record.get('prompt_position', 0)),
        tuple(int(v) for v in record.get('valid_labels', [])),
        str(record.get('prompt_score_version', PROMPT_SCORE_VERSION)),
    )


def validate_prompt_record(record, expected=None):
    required = {'prompt', 'concept_id', 'prompt_position', 'valid_labels'}
    missing = required - set(record)
    if missing:
        raise ValueError(f'A3 prompt record misses {sorted(missing)}.')
    version = record.get('prompt_score_version', PROMPT_SCORE_VERSION)
    if version != PROMPT_SCORE_VERSION:
        raise ValueError(f'Unsupported prompt score version: {version}')
    if expected is not None and prompt_identity(record) != prompt_identity(expected):
        raise ValueError(
            f'Prompt checkpoint identity mismatch: '
            f'{prompt_identity(record)} != {prompt_identity(expected)}')
    return record


def load_a3_records(root, expected_records=None):
    """Load an A3 directory or explicit pth without mutating the source."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'A3 checkpoint source does not exist: {root}')
    records = []
    if root.is_file():
        candidates = [root]
    else:
        candidates = sorted((root / 'prompt_best_params').glob('*.pth'))
        if not candidates:
            merged = root / 'independent_prompt_best_params.pth'
            if merged.exists():
                candidates = [merged]
    for path in candidates:
        payload = torch.load(path, map_location='cpu')
        if 'record' in payload:
            records.append(payload['record'])
        elif 'records' in payload:
            records.extend(payload['records'])
        elif path.name == 'independent_prompt_best_params.pth':
            # The merged legacy library has no identities; use the adjacent
            # JSON audit records and never invent them from tensor positions.
            result_json = path.with_name('prompt_optimization_results.json')
            if not result_json.exists():
                raise ValueError(
                    f'Merged A3 library lacks identity records: {path}')
            records.extend(json.loads(result_json.read_text(encoding='utf-8')))
        else:
            raise ValueError(f'Unrecognized A3 checkpoint structure: {path}')
    unique = {}
    for record in records:
        validate_prompt_record(record)
        key = prompt_identity(record)
        if key in unique:
            old = unique[key].get('parameters')
            new = record.get('parameters')
            if old != new:
                raise ValueError(f'Conflicting duplicate prompt record: {key}')
        unique[key] = record
    if expected_records is not None:
        expected = {prompt_identity(item): item for item in expected_records}
        for key, record in unique.items():
            if key not in expected:
                raise ValueError(f'Unexpected A3 prompt identity: {key}')
            validate_prompt_record(record, expected[key])
    return list(unique.values())


def implementation_fingerprint(paths):
    return {
        str(Path(path).resolve()): sha256_file(path)
        for path in paths if path and osp.isfile(path)
    }


def completed_matches(path, fingerprint):
    if not osp.isfile(path):
        return False
    with open(path, 'r', encoding='utf-8') as stream:
        marker = json.load(stream)
    return marker.get('fingerprint') == fingerprint
