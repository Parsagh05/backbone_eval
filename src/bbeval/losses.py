"""Prompt-fitting objective: focal + dice on the map, cross-entropy on the score."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import BackboneEvalConfig


def focal_loss(probability: torch.Tensor, target: torch.Tensor,
               gamma: float) -> torch.Tensor:
    """Multi-class focal loss over {normal, anomalous} at every pixel."""
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    truth = probability.gather(1, target[:, None])[:, 0]
    return -((1.0 - truth) ** gamma * truth.log()).mean()


def dice_loss(probability: torch.Tensor, target: torch.Tensor,
              eps: float = 1.0) -> torch.Tensor:
    intersection = (probability * target).sum(dim=(1, 2))
    cardinality = probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1.0 - (2.0 * intersection + eps) / (cardinality + eps)).mean()


def prompt_loss(config: BackboneEvalConfig, image_logits: torch.Tensor,
                pixel_logits: torch.Tensor, labels: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
    """AnomalyCLIP's objective.

    `image_loss + lam * sum_layers(focal + dice_abnormal + dice_normal)`, with
    `lam` as `pixel_loss_weight`. `pixel_logits` is [L, B, 2, h, w]; a plain
    [B, 2, h, w] map is accepted as a single layer.

    `loss_mode` selects which halves are used. "both" is AnomalyCLIP; "local"
    drops the image term, which is Tipsomaly's localisation-only ablation and
    leaves nothing opposing a collapse onto "normal everywhere".
    """
    total = image_logits.new_zeros(())
    if config.loss_mode in ("global", "both"):
        total = total + config.image_loss_weight * F.cross_entropy(image_logits, labels)
    if config.loss_mode in ("local", "both"):
        if pixel_logits.ndim == 4:
            pixel_logits = pixel_logits[None]
        target = F.interpolate(masks[:, None].float(),
                               size=pixel_logits.shape[-2:], mode="area") > 0.5
        target = target[:, 0].long()
        truth = target.float()
        pixel = total.new_zeros(())
        for layer_logits in pixel_logits:
            probability = layer_logits.softmax(dim=1)
            pixel = pixel + (focal_loss(probability, target, config.focal_gamma)
                             + dice_loss(probability[:, 1], truth)
                             + dice_loss(probability[:, 0], 1.0 - truth))
        total = total + config.pixel_loss_weight * pixel
    return total
