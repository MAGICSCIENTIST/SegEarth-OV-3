"""Prompt adapters and concept aggregators for CX-B."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


def _as_scalar_tensor(value, reference):
    return torch.as_tensor(value, device=reference.device,
                           dtype=reference.dtype)


def a3_probability(semantic, instance, presence, semantic_weight,
                   presence_gain, presence_intercept):
    """Exact identifiable-three A3 scoring rule."""
    alpha = _as_scalar_tensor(semantic_weight, semantic).clamp(EPS, 1 - EPS)
    gain = _as_scalar_tensor(presence_gain, semantic).clamp_min(EPS)
    intercept = _as_scalar_tensor(presence_intercept, semantic)
    fused = (alpha * semantic + (1.0 - alpha) * instance).clamp(
        EPS, 1.0 - EPS)
    presence = _as_scalar_tensor(presence, semantic)
    return torch.sigmoid(torch.logit(fused)
                         + gain * (presence - 0.5) + intercept)


class SegOV3PromptAdapter(nn.Module):
    """Original Eval LoveDA scoring, with no trainable prompt parameters."""

    def forward(self, heads):
        semantic = heads['semantic_score_map']
        instance = heads['instance_score_map']
        presence = _as_scalar_tensor(heads['presence_score'], semantic)
        # Also accept stacked heads [B,H,W] with one presence value per image.
        if presence.ndim == 1 and semantic.ndim >= 3:
            presence = presence.reshape(-1, *([1] * (semantic.ndim - 1)))
        return torch.maximum(semantic, instance) * presence


class A3PromptAdapter(nn.Module):
    """Three-scalar prompt adapter, optionally trainable."""

    def __init__(self, semantic_weight=0.5, presence_gain=1.0,
                 presence_intercept=0.0, trainable=False):
        super().__init__()
        alpha = torch.tensor(float(semantic_weight)).clamp(EPS, 1 - EPS)
        gain = torch.tensor(float(presence_gain)).clamp_min(EPS)
        values = (
            torch.logit(alpha),
            torch.log(torch.expm1(gain)),
            torch.tensor(float(presence_intercept)),
        )
        names = ('raw_semantic_weight', 'raw_presence_gain',
                 'presence_intercept')
        for name, value in zip(names, values):
            if trainable:
                self.register_parameter(name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)

    @property
    def semantic_weight(self):
        return torch.sigmoid(self.raw_semantic_weight)

    @property
    def presence_gain(self):
        return F.softplus(self.raw_presence_gain)

    def forward(self, heads):
        return a3_probability(
            heads['semantic_score_map'], heads['instance_score_map'],
            heads['presence_score'], self.semantic_weight,
            self.presence_gain, self.presence_intercept)

    def metadata(self):
        return dict(
            semantic_weight=float(self.semantic_weight.detach().cpu()),
            presence_gain=float(self.presence_gain.detach().cpu()),
            presence_intercept=float(self.presence_intercept.detach().cpu()))


class DynamicPromptHead(nn.Module):
    """Spatial alpha correction initialized to an A3 scalar response."""

    def __init__(self, semantic_weight=0.5, presence_gain=1.0,
                 presence_intercept=0.0, hidden_channels=16,
                 base_trainable=False):
        super().__init__()
        self.base = A3PromptAdapter(
            semantic_weight, presence_gain, presence_intercept,
            trainable=base_trainable)
        self.input_projection = nn.Conv2d(8, hidden_channels, 1)
        self.depthwise = nn.Conv2d(
            hidden_channels, hidden_channels, 3, padding=1,
            groups=hidden_channels)
        self.output_projection = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, heads):
        semantic = heads['semantic_score_map'].float()
        instance = heads['instance_score_map'].float()
        if semantic.ndim == 2:
            semantic = semantic[None, None]
            instance = instance[None, None]
            squeeze = True
        elif semantic.ndim == 3:
            semantic = semantic[:, None]
            instance = instance[:, None]
            squeeze = False
        else:
            squeeze = False
        disagreement = torch.abs(semantic - instance)
        local3 = F.avg_pool2d((semantic + instance) * 0.5, 3, 1, 1)
        local7 = F.avg_pool2d((semantic + instance) * 0.5, 7, 1, 3)
        presence = torch.as_tensor(
            heads['presence_score'], device=semantic.device,
            dtype=semantic.dtype).reshape(-1, 1, 1, 1)
        if presence.shape[0] == 1 and semantic.shape[0] != 1:
            presence = presence.expand(semantic.shape[0], -1, -1, -1)
        presence_map = presence.expand(-1, 1, *semantic.shape[-2:])
        features = torch.cat((
            semantic, instance, disagreement, local3, local7,
            semantic - local3, instance - local3, presence_map), dim=1)
        delta = self.output_projection(F.silu(self.depthwise(
            F.silu(self.input_projection(features)))))
        base_logit = torch.logit(self.base.semantic_weight.clamp(EPS, 1-EPS))
        alpha = torch.sigmoid(base_logit + delta[:, 0])
        fused = (alpha * semantic[:, 0]
                 + (1.0 - alpha) * instance[:, 0]).clamp(EPS, 1-EPS)
        gain = self.base.presence_gain
        output = torch.sigmoid(
            torch.logit(fused) + gain * (presence[:, 0] - 0.5)
            + self.base.presence_intercept)
        if squeeze:
            return output[0]
        return output


class ConceptAggregator(nn.Module):
    method = None

    def _validate(self, probabilities):
        if probabilities.ndim not in (3, 4):
            raise ValueError('probabilities must have shape [Q,H,W] or [B,Q,H,W].')
        if probabilities.shape[-3] < 1:
            raise ValueError('A concept must contain at least one prompt.')

    def forward(self, probabilities, presence=None, heads=None, regions=None,
                return_weights=False):
        self._validate(probabilities)
        # Every method is exactly identity for a single prompt, including
        # float64 tensors and exact zero/one endpoints.
        if probabilities.shape[-3] == 1:
            output = probabilities.select(-3, 0)
            weights = torch.ones_like(probabilities)
            return (output, weights) if return_weights else output
        output, weights = self.aggregate(
            probabilities, presence=presence, heads=heads, regions=regions)
        return (output, weights) if return_weights else output


class MeanAggregator(ConceptAggregator):
    method = 'A3-2'

    def aggregate(self, probabilities, **kwargs):
        return probabilities.mean(dim=-3), torch.full_like(
            probabilities, 1.0 / probabilities.shape[-3])


class MaxAggregator(ConceptAggregator):
    method = 'max'

    def aggregate(self, probabilities, **kwargs):
        output, indices = probabilities.max(dim=-3)
        weights = torch.zeros_like(probabilities).scatter_(
            -3, indices.unsqueeze(-3), 1.0)
        return output, weights


class LogSumExpAggregator(ConceptAggregator):
    method = 'logsumexp'

    def __init__(self, temperature=0.1, trainable=True):
        super().__init__()
        raw = torch.log(torch.expm1(torch.tensor(float(temperature))))
        if trainable:
            self.raw_temperature = nn.Parameter(raw)
        else:
            self.register_buffer('raw_temperature', raw)

    @property
    def temperature(self):
        return F.softplus(self.raw_temperature).clamp_min(EPS)

    def aggregate(self, probabilities, **kwargs):
        tau = self.temperature.to(probabilities)
        count = probabilities.shape[-3]
        output = tau * (
            torch.logsumexp(probabilities / tau, dim=-3) - math.log(count))
        weights = torch.softmax(probabilities / tau, dim=-3)
        return output.clamp(0.0, 1.0), weights


def _presence_map(probabilities, presence):
    if presence is None:
        return torch.zeros_like(probabilities)
    value = torch.as_tensor(presence, device=probabilities.device,
                            dtype=probabilities.dtype)
    if probabilities.ndim == 4:
        if value.ndim == 1:
            value = value.reshape(1, -1, 1, 1)
        elif value.ndim == 2:
            value = value[:, :, None, None]
    elif value.ndim == 1:
        value = value[:, None, None]
    while value.ndim < probabilities.ndim:
        value = value.unsqueeze(-1)
    return value.expand_as(probabilities)


class DynamicAggregator(ConceptAggregator):
    method = 'dynamic'

    def __init__(self, hidden_channels=16, feature_channels=8, prompt_count=1):
        super().__init__()
        self.feature_channels = feature_channels
        self.network = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 1), nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1))
        self.raw_response_scale = nn.Parameter(torch.tensor(0.0))
        self.prompt_bias = nn.Parameter(torch.zeros(int(prompt_count)))

    def features(self, probabilities, presence=None, heads=None, regions=None):
        mean = probabilities.mean(dim=-3, keepdim=True)
        maximum = probabilities.max(dim=-3, keepdim=True).values
        minimum = probabilities.min(dim=-3, keepdim=True).values
        std = probabilities.std(dim=-3, keepdim=True, unbiased=False)
        expand = lambda x: x.expand_as(probabilities)
        return torch.stack((
            probabilities, expand(mean), expand(maximum), expand(minimum),
            expand(std), probabilities - expand(mean),
            _presence_map(probabilities, presence),
            torch.ones_like(probabilities)), dim=-3)

    def aggregate(self, probabilities, presence=None, heads=None, regions=None):
        original_batched = probabilities.ndim == 4
        if not original_batched:
            probabilities = probabilities.unsqueeze(0)
        features = self.features(probabilities, presence, heads, regions)
        b, q, f, h, w = features.shape
        logits = self.network(features.reshape(b*q, f, h, w)).reshape(b,q,h,w)
        logits = logits + F.softplus(self.raw_response_scale) * probabilities
        if self.prompt_bias.numel() not in (1, q):
            raise ValueError('prompt_bias must be scalar or have Q entries.')
        bias = self.prompt_bias
        if bias.numel() == q:
            bias = bias - bias.mean()
            logits = logits + bias.reshape(1, q, 1, 1)
        weights = torch.softmax(logits, dim=1)
        output = (weights * probabilities).sum(dim=1)
        if not original_batched:
            return output[0], weights[0]
        return output, weights


def _head_tensor(heads, key, probabilities):
    if not heads or key not in heads:
        return torch.zeros_like(probabilities)
    value = heads[key].to(probabilities)
    if value.ndim == probabilities.ndim - 1:
        value = value.unsqueeze(0)
    return value.expand_as(probabilities)


class ReliabilityDynamicAggregator(DynamicAggregator):
    method = 'reliability_dynamic'

    def __init__(self, hidden_channels=16, prompt_count=1):
        super().__init__(hidden_channels, feature_channels=11,
                         prompt_count=prompt_count)

    def features(self, probabilities, presence=None, heads=None, regions=None):
        base = super().features(probabilities, presence, heads, regions)
        semantic = _head_tensor(heads, 'semantic_score_map', probabilities)
        instance = _head_tensor(heads, 'instance_score_map', probabilities)
        return torch.cat((base, torch.stack((
            semantic, instance, torch.abs(semantic-instance)), dim=-3)), dim=-3)


class RegionDynamicAggregator(DynamicAggregator):
    method = 'region_dynamic'

    def __init__(self, hidden_channels=16, prompt_count=1):
        super().__init__(hidden_channels, feature_channels=11,
                         prompt_count=prompt_count)
        self.network = nn.Sequential(
            nn.Conv2d(11, hidden_channels, 1), nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                      groups=hidden_channels), nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1))

    def features(self, probabilities, presence=None, heads=None, regions=None):
        base = super().features(probabilities, presence, heads, regions)
        b, q, h, w = probabilities.shape
        flat = probabilities.reshape(b*q, 1, h, w)
        local3 = F.avg_pool2d(flat, 3, 1, 1).reshape(b,q,h,w)
        local7 = F.avg_pool2d(flat, 7, 1, 3).reshape(b,q,h,w)
        gradient = torch.abs(local3-local7)
        return torch.cat((base, torch.stack(
            (local3, local7, gradient), dim=-3)), dim=-3)


def region_support(probabilities, regions):
    support = torch.zeros_like(probabilities)
    if regions is None:
        return support
    # regions is one (masks, quality) item per prompt.
    for prompt_idx, item in enumerate(regions):
        if item is None:
            continue
        masks, quality = item
        masks = masks.to(probabilities).float()
        quality = quality.to(probabilities).float()
        if masks.numel() == 0:
            continue
        weighted = masks * quality.reshape(-1, 1, 1)
        merged = weighted.sum(0) / masks.sum(0).clamp_min(1.0)
        if probabilities.ndim == 4:
            support[:, prompt_idx] = merged
        else:
            support[prompt_idx] = merged
    return support.clamp(0.0, 1.0)


class SAMRegionDynamicAggregator(DynamicAggregator):
    method = 'sam_region_dynamic'

    def __init__(self, hidden_channels=16, prompt_count=1):
        super().__init__(hidden_channels, feature_channels=11,
                         prompt_count=prompt_count)

    def features(self, probabilities, presence=None, heads=None, regions=None):
        base = super().features(probabilities, presence, heads, regions)
        support = region_support(probabilities, regions)
        covered = (support > 0).to(probabilities)
        disagreement = torch.abs(probabilities-support) * covered
        return torch.cat((base, torch.stack(
            (support, covered, disagreement), dim=-3)), dim=-3)


AGGREGATORS = {
    'A3-2': MeanAggregator,
    'max': MaxAggregator,
    'logsumexp': LogSumExpAggregator,
    'dynamic': DynamicAggregator,
    'reliability_dynamic': ReliabilityDynamicAggregator,
    'region_dynamic': RegionDynamicAggregator,
    'sam_region_dynamic': SAMRegionDynamicAggregator,
}


def build_prompt_module(method, parameters, **kwargs):
    if method == 'segov3':
        return SegOV3PromptAdapter()
    if method == 'A3-2':
        return A3PromptAdapter(**parameters, trainable=kwargs.get('trainable', False))
    if method == 'dynamic_head':
        return DynamicPromptHead(**parameters,
                                 hidden_channels=kwargs.get('hidden_channels', 16),
                                 base_trainable=kwargs.get(
                                     'base_trainable', False))
    raise ValueError(f'Unsupported prompt method: {method}')


def build_concept_aggregator(method, prompt_count, **kwargs):
    if prompt_count < 1:
        raise ValueError('prompt_count must be positive.')
    if prompt_count == 1:
        return MeanAggregator()
    if method not in AGGREGATORS:
        raise ValueError(f'Unsupported concept method: {method}')
    if method == 'logsumexp':
        return LogSumExpAggregator(
            temperature=kwargs.get('temperature', 0.1),
            trainable=kwargs.get('trainable', True))
    if method in (
            'dynamic', 'reliability_dynamic', 'region_dynamic',
            'sam_region_dynamic'):
        return AGGREGATORS[method](
            hidden_channels=kwargs.get('hidden_channels', 16),
            prompt_count=prompt_count)
    return AGGREGATORS[method]()


def save_variant(path, module, metadata, training_state=None):
    from .checkpoints import atomic_torch_save
    payload = dict(
        format='cx_b_variant_v1', module_type=module.__class__.__name__,
        method=metadata.get('method'), metadata=dict(metadata),
        state_dict=module.state_dict())
    if training_state is not None:
        payload['training_state'] = training_state
    atomic_torch_save(payload, path)


def load_variant(path, device='cpu', payload=None):
    checkpoint = torch.load(path, map_location=device) if payload is None else payload
    if checkpoint.get('format') != 'cx_b_variant_v1':
        raise ValueError(f'Unsupported CX-B checkpoint: {path}')
    metadata = checkpoint['metadata']
    kind = metadata.get('kind')
    if kind == 'prompt':
        module = build_prompt_module(
            metadata['method'], metadata['base_parameters'],
            hidden_channels=metadata.get('hidden_channels', 16),
            base_trainable=metadata.get('base_trainable', False))
    elif kind == 'concept':
        module = build_concept_aggregator(
            metadata['method'], metadata['prompt_count'],
            hidden_channels=metadata.get('hidden_channels', 16),
            temperature=metadata.get('temperature', 0.1))
    else:
        raise ValueError(f'Unknown variant kind: {kind!r}')
    module.load_state_dict(checkpoint['state_dict'])
    module.to(device).eval()
    return module, metadata
