"""Independent binary objectives used by both CX-B training stages."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IndependentBinaryLoss(nn.Module):
    """Compute one binary objective without reading other concepts."""

    def __init__(self, method='A3-2', soft_iou_weight=1.0,
                 recall_weight=0.1, recall_target=0.3, bce_weight=1.0,
                 positive_weight=1.0, tversky_weight=1.0,
                 tversky_alpha_fp=0.6, tversky_beta_fn=0.4, eps=1e-6):
        super().__init__()
        if method not in ('A3-2', 'soft_iou_bce', 'bce_tversky'):
            raise ValueError(f'Unsupported loss method: {method}')
        self.method = method
        self.soft_iou_weight = float(soft_iou_weight)
        self.recall_weight = float(recall_weight)
        self.recall_target = float(recall_target)
        self.bce_weight = float(bce_weight)
        self.positive_weight = float(positive_weight)
        self.tversky_weight = float(tversky_weight)
        self.alpha = float(tversky_alpha_fp)
        self.beta = float(tversky_beta_fn)
        self.eps = float(eps)

    def forward(self, probability, target, valid_mask=None):
        probability = probability.float()
        target = target.to(probability.device, probability.dtype)
        if valid_mask is None:
            valid_mask = torch.ones_like(target, dtype=torch.bool)
        valid_mask = valid_mask.to(probability.device, torch.bool)
        if not torch.any(valid_mask):
            return None
        p = probability[valid_mask].clamp(self.eps, 1.0 - self.eps)
        y = target[valid_mask]
        intersection = (p * y).sum()
        pred_mass = p.sum()
        target_mass = y.sum()
        fp = (p * (1.0 - y)).sum()
        fn = ((1.0 - p) * y).sum()

        # Soft IoU alone is almost flat for an empty target.  Use an explicit
        # false-positive mass loss so every loss family suppresses hallucinated
        # responses instead of treating the sample as a no-op.
        if target_mass.item() == 0:
            return p.mean()

        soft_iou = (intersection + self.eps) / (
            pred_mass + target_mass - intersection + self.eps)
        soft_iou_loss = 1.0 - soft_iou
        recall = ((intersection + self.eps) / (target_mass + self.eps)
                  if target_mass > 0 else p.new_tensor(1.0))
        recall_loss = F.relu(p.new_tensor(self.recall_target) - recall)
        positive_weights = torch.where(
            y > 0.5, p.new_tensor(self.positive_weight), p.new_tensor(1.0))
        bce = F.binary_cross_entropy(p, y, weight=positive_weights)
        tversky = (intersection + self.eps) / (
            intersection + self.alpha * fp + self.beta * fn + self.eps)

        if self.method == 'A3-2':
            return (self.soft_iou_weight * soft_iou_loss
                    + self.recall_weight * recall_loss)
        if self.method == 'soft_iou_bce':
            return (self.soft_iou_weight * soft_iou_loss
                    + self.bce_weight * bce)
        return self.bce_weight * bce + self.tversky_weight * (1.0 - tversky)


def build_binary_loss(config, stage):
    cx = config['cx_b'] if 'cx_b' in config else config
    settings = dict(cx.get('loss', {}).get(stage, {}))
    settings.setdefault('method', cx.get('loss_method', 'A3-2'))
    return IndependentBinaryLoss(**settings)
