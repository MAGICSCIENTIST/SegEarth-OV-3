"""Raw-head providers and online SAM region extraction."""

from __future__ import annotations

import contextlib
import os.path as osp

import torch
import torch.nn.functional as F
from PIL import Image


def export_sam_region_cache(sam_outputs, sample_meta, cache_cfg):
    """Reserved API. Region output caching is intentionally prohibited."""
    pass


def validate_cache_contract(data_mode, sam_region_cfg):
    if data_mode not in ('cache', 'online_sam'):
        raise ValueError(f'Unknown head data mode: {data_mode}')
    if sam_region_cfg.get('read_output_cache') or sam_region_cfg.get(
            'write_output_cache'):
        raise ValueError('CX-B forbids SAM region/head output caches.')
    if data_mode == 'online_sam' and sam_region_cfg.get('head_cache'):
        raise ValueError('online_sam cannot read a disk head cache.')


class CachedHeadProvider:
    def __init__(self, cache_root, model):
        self.cache_root = osp.abspath(cache_root)
        self.model = model

    def load(self, split, prompt_idx, sample_idx, image_path):
        import train_prompt_params_independent as original
        path = original.cache_file(
            self.cache_root, split, prompt_idx, sample_idx, image_path)
        return original.load_heads_cpu(
            path, image_path, prompt_idx,
            self.model.query_words[prompt_idx],
            original.cache_signature(self.model))


def _autocast(device, amp_dtype):
    if device.type != 'cuda' or amp_dtype == 'float32':
        return contextlib.nullcontext()
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[amp_dtype]
    return torch.autocast(device_type='cuda', dtype=dtype)


class OnlineSAMHeadProvider:
    """Run SAM for every request; share only the current image encoding."""

    def __init__(self, model, config):
        self.model = model
        self.config = dict(config)
        validate_cache_contract('online_sam', self.config)
        if model.device.type == 'cpu' and self.config.get(
                'amp_dtype', 'bfloat16') != 'float32':
            self.config['amp_dtype'] = 'float32'

    def image_outputs(self, image_path, prompt_indices):
        image = Image.open(image_path).convert('RGB')
        crop = int(getattr(self.model, 'slide_crop', 0) or 0)
        if crop > 0 and (image.width > crop or image.height > crop):
            raise ValueError(
                f'online_sam input {image.size} exceeds slide_crop={crop}; '
                'silent cropping is forbidden.')
        outputs = {}
        with torch.no_grad(), _autocast(
                self.model.device, self.config.get('amp_dtype', 'bfloat16')):
            state = self.model.processor.set_image(image)
            for prompt_idx in prompt_indices:
                state, heads = self.model._extract_prompt_heads_from_state(
                    state, prompt_idx, image.width, image.height)
                masks = state.get('masks_logits')
                if masks is None:
                    masks = torch.empty(
                        (0, image.height, image.width), device=self.model.device)
                masks = masks.detach().float()
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                elif masks.ndim == 2:
                    masks = masks.unsqueeze(0)
                if masks.ndim != 3:
                    raise ValueError(
                        f'Unexpected SAM mask shape: {tuple(masks.shape)}')
                quality_key = self.config.get('quality_source', 'object_score')
                quality = state.get(quality_key)
                if quality is None:
                    quality = torch.empty((0,), device=self.model.device)
                quality = quality.detach().float().reshape(-1)
                keep = quality >= float(self.config.get('min_region_score', 0.0))
                masks, quality = masks[keep], quality[keep]
                limit = int(self.config.get('max_masks_per_prompt', 64))
                if limit > 0 and quality.numel() > limit:
                    order = torch.argsort(quality, descending=True)[:limit]
                    masks, quality = masks[order], quality[order]
                threshold = float(self.config.get('mask_threshold', 0.5))
                # masks_logits is already sigmoid-normalized by this processor.
                masks = (masks >= threshold).detach()
                outputs[prompt_idx] = dict(
                    heads={key: value.detach().float()
                           for key, value in heads.items()},
                    regions=(masks, quality))
        return outputs
