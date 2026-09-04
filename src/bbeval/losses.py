"""Prompt-fitting objective: focal + dice on the map, cross-entropy on the score."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import BackboneEvalConfig


def focal_loss(probability: torch.Tensor, target: torch.Tensor,
               gamma: float, smooth: float = 1e-5) -> torch.Tensor:
    """AnomalyCLIP's focal loss over two-channel pixel probabilities."""
    one_hot = F.one_hot(target.long(), num_classes=2).permute(0, 3, 1, 2)
    one_hot = one_hot.to(device=probability.device, dtype=probability.dtype)
    one_hot = one_hot.clamp(smooth, 1.0 - smooth)
    truth = (one_hot * probability).sum(dim=1) + smooth
    truth = truth.clamp_min(smooth)
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
        # Upsample the prediction to the mask, as AnomalyCLIP does, rather than
        # shrinking the mask to the patch grid. Downsampling a 64x64 mask to a
        # 24x24 grid loses small defects outright, and it also trained against a
        # different target than the metrics are computed on.
        target = (masks > 0.5).long()
        truth = target.float()
        pixel = total.new_zeros(())
        for layer_logits in pixel_logits:
            probability = layer_logits.softmax(dim=1)
            if probability.shape[-2:] != target.shape[-2:]:
                # Bilinear on a softmax output keeps the two channels summing to
                # one, so the interpolated map is still a distribution.
                probability = F.interpolate(probability, size=target.shape[-2:],
                                            mode="bilinear", align_corners=False)
            pixel = pixel + (focal_loss(probability, target, config.focal_gamma,
                                        config.focal_smooth)
                             + dice_loss(probability[:, 1], truth)
                             + dice_loss(probability[:, 0], 1.0 - truth))
        total = total + config.pixel_loss_weight * pixel
    return total
