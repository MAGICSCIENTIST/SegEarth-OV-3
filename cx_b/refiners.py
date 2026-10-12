"""Vocabulary-independent score refinement (Deep Sets / CAT-Seg motivation).

No class-indexed parameters. A disabled refiner is represented by None, so the
legacy path neither constructs a network nor consumes random numbers.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .configuration import refiner_settings

SCHEMA = 'probability_own_local3_local7_other_mean_max_margins_background_v1'
JOINT_FORMATS = ('cx_b_joint_v1', 'cx_b_joint_v2')


class OVClassSpatialRefiner(nn.Module):
    method = 'ov_class_spatial'

    def __init__(self, hidden_channels=16, strength=1.0):
        super().__init__()
        if not isinstance(hidden_channels, int) or hidden_channels < 1:
            raise ValueError('refiner hidden_channels must be a positive integer')
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('refiner strength must be in [0,1]')
        self.hidden_channels, self.strength = hidden_channels, float(strength)
        self.network = nn.Sequential(
            nn.Conv2d(8, hidden_channels, 1), nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
            nn.SiLU(), nn.Conv2d(hidden_channels, 1, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, scores, background=0):
        if self.strength == 0:
            return scores
        if scores.ndim not in (3, 4) or scores.shape[-3] < 1:
            raise ValueError('refiner expects nonempty [C,H,W] or [B,C,H,W] scores')
        if not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('refiner expects finite probabilities in [0,1]')
        if background is not None and not 0 <= background < scores.shape[-3]:
            raise ValueError('background role must identify an input class or be None')
        unbatched = scores.ndim == 3
        s = scores.unsqueeze(0) if unbatched else scores
        # Keep odds arithmetic in at least float32; preserve float64 for checks.
        s = s if s.dtype == torch.float64 else s.float()
        b, c, h, w = s.shape
        local = s.reshape(b*c, 1, h, w)
        local3 = F.avg_pool2d(local, 3, 1, 1).reshape_as(s)
        local7 = F.avg_pool2d(local, 7, 1, 3).reshape_as(s)
        if c == 1:
            mean = maximum = torch.zeros_like(s)
        else:
            mean = (s.sum(1, keepdim=True)-s)/(c-1)
            top, indices = s.topk(2, dim=1)
            ids = torch.arange(c, device=s.device).view(1,c,1,1)
            maximum = torch.where(ids == indices[:, :1], top[:, 1:2], top[:, :1])
        role = torch.zeros_like(s)
        if background is not None:
            role[:, background] = 1
        features = torch.stack((s, local3, local7, mean, maximum, s-mean, s-maximum, role), dim=2)
        raw = self.network(features.reshape(b*c, 8, h, w)).reshape_as(s)
        delta = self.strength * torch.tanh(raw)
        t = torch.expm1(delta)
        refined = (s*(1+t)/(1+s*t)).clamp(0, 1)
        refined = refined.to(scores.dtype)
        return refined[0] if unbatched else refined


def build_refiner(cx, device='cpu'):
    method, strength = refiner_settings(cx)
    if method == 'none':
        return None
    return OVClassSpatialRefiner(cx.get('refiner_hidden_channels', 16), strength).to(device)


def refiner_payload(module, **provenance):
    return dict(format='cx_b_refiner_v1', method=module.method, feature_schema=SCHEMA,
                hidden_channels=module.hidden_channels, strength=module.strength,
                state_dict={k: v.detach().cpu().clone() for k,v in module.state_dict().items()},
                provenance=provenance)


def load_refiner(payload_or_path, device='cpu', strength=None):
    """Portable refiner only: class names/count are provenance, never parameters."""
    payload = (payload_or_path if isinstance(payload_or_path, dict)
               else torch.load(payload_or_path, map_location='cpu'))
    if payload.get('format') != 'cx_b_refiner_v1' or payload.get('method') != 'ov_class_spatial':
        raise ValueError('Invalid standalone refiner checkpoint')
    if payload.get('feature_schema') != SCHEMA:
        raise ValueError('Refiner feature_schema mismatch')
    value = payload['strength'] if strength is None else strength
    # Loading must not perturb the caller's subsequent initialization/order.
    with torch.random.fork_rng(devices=[]):
        module = OVClassSpatialRefiner(payload['hidden_channels'], value)
    module.load_state_dict(payload['state_dict'], strict=True)
    return module.to(device).eval()


def validate_joint_refiner(bundle):
    if bundle.get('format') not in JOINT_FORMATS:
        raise ValueError('Invalid joint checkpoint format')
    if bundle['format'] == 'cx_b_joint_v2':
        if not bundle.get('refiner') or not bundle.get('base_metrics'):
            raise ValueError('Joint v2 checkpoint missing refiner/base_metrics')
        payload = bundle['refiner']
        if payload.get('feature_schema') != SCHEMA:
            raise ValueError('Joint refiner feature_schema mismatch')
    elif bundle.get('refiner') is not None:
        raise ValueError('A refiner requires joint checkpoint v2')


class ConceptModules(dict):
    """Preserve the existing six-value load_run API, with optional full head."""
    refiner = None
    background = 0
    raw_scores = None


def evaluation_refiner_options(args):
    options = dict(joint_refiner=getattr(args, 'joint_refiner', None),
                   refiner_strength=getattr(args, 'joint_refiner_strength', None))
    return {key: value for key, value in options.items() if value is not None}


def ablation_folder(args):
    options = evaluation_refiner_options(args)
    method, strength = options.get('joint_refiner'), options.get('refiner_strength')
    if method is None and strength is None:
        return None
    refiner_settings(dict(joint_refiner=method or 'ov_class_spatial',
                         joint_refiner_strength=1. if strength is None else strength))
    return f'refiner_{method or "saved"}__strength_{"saved" if strength is None else format(strength, "g")}'


def predict_open_vocabulary(heads_by_concept, refiner=None, background=None):
    """Query-independent SegOV3/max prior -> optional shared refiner.

    The caller supplies actual SAM heads for EACH requested prompt, grouped by
    concept. Never reassign trained per-prompt A3 weights to unseen queries.
    Changing this prior from an A3 run is a domain change, not a reproduced F1.
    """
    from .modules import SegOV3PromptAdapter
    if not heads_by_concept or any(not group for group in heads_by_concept):
        raise ValueError('Each query concept requires at least one actual prompt head')
    adapter = SegOV3PromptAdapter()
    scores = torch.stack([torch.stack([adapter(h) for h in group]).max(0).values
                          for group in heads_by_concept])
    return scores if refiner is None else refiner(scores, background=background)
