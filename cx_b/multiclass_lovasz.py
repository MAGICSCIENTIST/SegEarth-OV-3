"""Multiclass Lovasz-Softmax, averaged over classes present in valid targets.

Implements the sorted Jaccard-error surrogate from Berman et al., CVPR 2018.
Input is logits (not independent sigmoid probabilities); ignored pixels are
removed before sorting. Keep this small helper identical in MTP and CX-B.
"""

import torch


def lovasz_softmax(logits, target, ignore_index=255):
    """Accept [B,C,H,W]/[B,H,W], or one [C,H,W]/[H,W] image.

    Computations use float32 even under autocast. An all-ignored target returns
    a differentiable zero. Absent classes are excluded from the class average,
    but still compete through the softmax denominator.
    """
    if logits.ndim == 3:
        logits, target = logits.unsqueeze(0), target.unsqueeze(0)
    if logits.ndim != 4 or target.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError('Lovasz expects logits [B,C,H,W] and targets [B,H,W].')
    scores = logits.float()
    labels = target.reshape(-1)
    valid = labels != ignore_index
    labels = labels[valid]
    if labels.numel() == 0:
        return scores.sum() * 0.0
    if (labels < 0).any() or (labels >= scores.shape[1]).any():
        raise ValueError('Lovasz target contains an out-of-range class index.')
    probabilities = scores.softmax(dim=1).movedim(1, -1).reshape(-1, scores.shape[1])[valid]
    losses = []
    for class_index in labels.unique():
        foreground = (labels == class_index).float()
        errors = (foreground - probabilities[:, class_index]).abs()
        sorted_errors, order = errors.sort(descending=True)
        sorted_foreground = foreground[order]
        positives = sorted_foreground.sum()
        intersection = positives - sorted_foreground.cumsum(0)
        union = positives + (1.0 - sorted_foreground).cumsum(0)
        jaccard = 1.0 - intersection / union
        gradient = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
        losses.append(torch.dot(sorted_errors, gradient))
    return torch.stack(losses).mean()
