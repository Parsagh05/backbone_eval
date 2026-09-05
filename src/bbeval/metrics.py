"""Pixel and image metrics (pptx slide 21), plus the slide-24 calibration term."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

PIXEL_METRICS = ("pixel_auroc", "pixel_f1max", "pixel_aupro", "pixel_threshold")
IMAGE_METRICS = ("image_auroc", "image_f1max", "image_ap", "image_threshold")
# Reported separately so ranking tables that expect the six classic metrics
# stay unchanged.
CALIBRATION_METRICS = ("image_ece",)
ALL_METRICS = PIXEL_METRICS + IMAGE_METRICS


def _f1_max_and_threshold(truth, score) -> tuple[float, float]:
    """F1-max and the threshold that attains it."""
    precision, recall, thresholds = precision_recall_curve(truth, score)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    best = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    threshold = float(thresholds[best]) if len(thresholds) else float("nan")
    return float(f1[best]), threshold


def expected_calibration_error(labels, scores, n_bins: int = 15) -> float:
    """Binary ECE (Guo et al., ICML 2017) -- pptx slide 24."""
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")
    # Map raw anomaly scores to [0, 1] via a rank-preserving min-max.
    low, high = float(scores.min()), float(scores.max())
    probs = (np.full_like(scores, 0.5) if high <= low
             else (scores - low) / (high - low))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = labels.size
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probs >= left) & (probs < right if right < 1.0 else probs <= right)
        if not mask.any():
            continue
        accuracy = float(labels[mask].mean())
        confidence = float(probs[mask].mean())
        ece += (mask.sum() / total) * abs(accuracy - confidence)
    return float(ece)


def image_metrics(labels, scores) -> dict[str, float]:
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.min() == labels.max():          # single-class: undefined, not zero
        return dict.fromkeys(IMAGE_METRICS, float("nan"))
    f1, threshold = _f1_max_and_threshold(labels, scores)
    return {"image_auroc": float(roc_auc_score(labels, scores)),
            "image_f1max": f1,
            "image_ap": float(average_precision_score(labels, scores)),
            "image_threshold": threshold}


def compute_aupro(masks, maps, max_fpr: float = 0.30,
                  num_thresholds: int = 200) -> float:
    """Region-averaged overlap vs. FPR, integrated to `max_fpr` and normalised."""
    masks = np.asarray(masks).astype(bool)
    if not masks.any() or masks.all():
        return float("nan")

    regions = []
    for index in range(masks.shape[0]):
        if not masks[index].any():
            continue
        labelled = measure.label(masks[index], connectivity=2)
        for region in measure.regionprops(labelled):
            regions.append((index, tuple(region.coords.T), float(region.area)))
    if not regions:
        return float("nan")

    normal = ~masks
    normal_total = float(normal.sum())
    lowest, highest = float(maps.min()), float(maps.max())
    if not np.isfinite(lowest) or highest <= lowest:
        return float("nan")
    thresholds = np.linspace(highest, lowest, num_thresholds)

    pro, fpr = [], []
    for threshold in thresholds:
        predicted = maps >= threshold
        pro.append(float(np.mean([predicted[index][coordinates].sum() / area
                                  for index, coordinates, area in regions])))
        fpr.append(float((predicted & normal).sum() / normal_total))

    fpr, pro = np.asarray(fpr), np.asarray(pro)
    order = np.argsort(fpr)
    fpr, pro = fpr[order], pro[order]
    if fpr[0] > 0:                                # anchor the curve at the origin
        fpr, pro = np.concatenate([[0.0], fpr]), np.concatenate([[0.0], pro])
    # Interpolating onto a fixed grid keeps the integral well defined even when a
    # saturated map crosses max_fpr at its very first threshold -- that deserves
    # an AUPRO near zero, not a NaN.
    grid = np.linspace(0.0, max_fpr, num_thresholds)
    return float(_trapezoid(np.interp(grid, fpr, pro), grid) / max_fpr)


def pixel_metrics(masks, maps, with_aupro: bool = True,
                  max_fpr: float = 0.30, num_thresholds: int = 200) -> dict[str, float]:
    masks = np.asarray(masks).astype(np.uint8)
    maps = np.asarray(maps, dtype=np.float32)
    flat_truth, flat_score = masks.ravel(), maps.ravel().astype(np.float64)
    if flat_truth.min() == flat_truth.max():
        return dict.fromkeys(PIXEL_METRICS, float("nan"))
    f1, threshold = _f1_max_and_threshold(flat_truth, flat_score)
    aupro = (compute_aupro(masks, maps, max_fpr, num_thresholds)
             if with_aupro else float("nan"))
    return {"pixel_auroc": float(roc_auc_score(flat_truth, flat_score)),
            "pixel_f1max": f1,
            "pixel_aupro": aupro,
            "pixel_threshold": threshold}


def evaluate_shard(masks, maps, labels, scores, with_aupro: bool = True,
                   with_ece: bool = True, max_fpr: float = 0.30,
                   num_thresholds: int = 200, ece_bins: int = 15) -> dict[str, float]:
    result = {**pixel_metrics(masks, maps, with_aupro, max_fpr, num_thresholds),
              **image_metrics(labels, scores)}
    if with_ece:
        result["image_ece"] = expected_calibration_error(labels, scores, ece_bins)
    return result


def resize_masks(masks, size: int) -> np.ndarray:
    """Area-downsample binary masks, keeping defects too small to survive it.

    Used only for *storage*: metrics are computed against the full-resolution
    masks during the sweep.
    """
    if masks.shape[-1] == size:
        return np.asarray(masks, dtype=np.uint8)
    tensor = torch.from_numpy(np.asarray(masks, dtype=np.float32))[:, None]
    reduced = F.interpolate(tensor, size=(size, size), mode="area") > 0.5
    empty = reduced.flatten(1).any(dim=1).logical_not() & tensor.flatten(1).any(dim=1)
    if empty.any():
        rescued = F.adaptive_max_pool2d(tensor[empty], size) > 0
        reduced[empty] = rescued
    return reduced[:, 0].numpy().astype(np.uint8)


def resize_maps(maps, size: int) -> np.ndarray:
    """Bilinear resampling of stored low-res maps, for higher-resolution scoring."""
    if maps.shape[-1] == size:
        return np.asarray(maps, dtype=np.float32)
    tensor = torch.from_numpy(np.asarray(maps, dtype=np.float32))[:, None]
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear",
                            align_corners=False)
    return resized[:, 0].numpy()
