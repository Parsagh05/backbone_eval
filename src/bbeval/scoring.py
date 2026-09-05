"""Turning patch and text embeddings into anomaly scores and maps.

Patch and text embeddings are compared by cosine similarity, scaled by
`logit_scale_for` and softmaxed over {normal, anomalous}; the anomalous channel
is the anomaly map.

Note on SigLIP2: a two-class softmax over `t*cos + b` is invariant to the
learned bias `b`, which cancels between the two prompts. The softmax score is
therefore well defined for a sigmoid-trained model, and the ranking metrics
cannot depend on `b`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .backbones import Backbone
from .config import BackboneEvalConfig


def logit_scale_for(config: BackboneEvalConfig, backbone: Backbone) -> float:
    """The scale applied to every cosine, for training and inference alike.

    Defaults to AnomalyCLIP's 1/0.07 rather than the backbone's learned scale
    (CLIP 100, SigLIP2 ~108), which saturates the two-class softmax and flattens
    the focal gradient. Ranking metrics are invariant to it, but the training
    signal is not. The optional score-fusion ablation is also scale-sensitive.
    """
    temperature = (config.map_temperature if config.map_temperature is not None
                   else backbone.temperature)
    return 1.0 / temperature


def dense_logits_per_layer(dense: Mapping[int, torch.Tensor], text: torch.Tensor,
                           logit_scale: float) -> torch.Tensor:
    """Patch-text logits for each selected block -> [L, B, 2, h, w].

    AnomalyCLIP supervises every layer separately, so training needs the stack
    rather than the average.
    """
    per_layer = []
    for _, tokens in sorted(dense.items()):
        normed = F.normalize(tokens.float(), dim=-1)
        per_layer.append(logit_scale * torch.einsum("bhwd,kd->bkhw", normed, text))
    return torch.stack(per_layer)


def global_logits(features: Mapping[str, Any], text: torch.Tensor,
                  logit_scale: float, config: BackboneEvalConfig,
                  has_two_tokens: bool) -> torch.Tensor:
    token = config.global_token if has_two_tokens else "object"
    normed = F.normalize(features[token].float(), dim=-1)
    return logit_scale * normed @ text.t()               # [B, 2]


def _gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """SciPy-compatible Gaussian smoothing used by AnomalyCLIP at inference."""
    if not sigma:
        return maps
    # scipy.ndimage.gaussian_filter defaults to truncate=4.0. Reflect padding
    # is its default boundary treatment; cap only for tiny synthetic test maps.
    radius = max(1, int(4.0 * sigma + 0.5))
    radius = min(radius, maps.shape[-2] - 1, maps.shape[-1] - 1)
    grid = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    kernel = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    maps = F.pad(maps, (radius, radius, 0, 0), mode="reflect")
    maps = F.conv2d(maps, kernel.view(1, 1, 1, -1))
    maps = F.pad(maps, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(maps, kernel.view(1, 1, -1, 1))


def to_anomaly_map(logits: torch.Tensor, config: BackboneEvalConfig,
                   map_res: int | None = None) -> torch.Tensor:
    """AnomalyCLIP map: sum per-layer probabilities after spatial upsampling.

    Accepts [L,B,2,h,w], or [B,2,h,w] for a single-layer control. Official
    AnomalyCLIP softmaxes each layer separately, upsamples its probability map,
    and sums the maps; averaging logits before softmax is not equivalent.
    """
    size = map_res or config.map_res
    if logits.ndim == 4:
        logits = logits[None]
    if logits.ndim != 5:
        raise ValueError(f"expected [L,B,2,h,w] logits, got {tuple(logits.shape)}")
    layers, batch = logits.shape[:2]
    probability = logits.softmax(dim=2)[:, :, 1:2]
    probability = F.interpolate(
        probability.reshape(layers * batch, 1, *probability.shape[-2:]),
        size=(size, size), mode="bilinear", align_corners=False)
    probability = probability.reshape(layers, batch, 1, size, size).sum(dim=0)
    return _gaussian_blur(probability, config.gaussian_sigma)[:, 0]


def peak_evidence(maps: torch.Tensor, config: BackboneEvalConfig) -> torch.Tensor:
    """The strongest local evidence: max(), or a top-k mean if configured."""
    flat = maps.flatten(1)
    if config.topk_fraction and config.topk_fraction > 0:
        k = max(1, int(round(config.topk_fraction * flat.shape[1])))
        return flat.topk(k, dim=1).values.mean(dim=1)
    return flat.max(dim=1).values


@torch.no_grad()
def anomaly_outputs(config: BackboneEvalConfig, backbone: Backbone,
                    texts: Mapping[str, torch.Tensor], images_uint8: torch.Tensor,
                    map_res: int | None = None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{mode: (scores [B], maps [B,M,M])} for every configured prompt mode.

    Every mode comes out of a single visual forward pass, so measuring all of
    them costs almost nothing over measuring one.
    """
    logit_scale = logit_scale_for(config, backbone)
    with torch.autocast("cuda", enabled=(config.amp and config.device == "cuda")):
        features = backbone.encode(backbone.preprocess(images_uint8))

    maps, global_probability = {}, {}
    for key, text in texts.items():
        maps[key] = to_anomaly_map(
            dense_logits_per_layer(features["dense"], text, logit_scale),
            config, map_res)
        global_probability[key] = global_logits(
            features, text, logit_scale, config,
            backbone.has_two_global_tokens).softmax(-1)[:, 1]

    def score(mode: str) -> torch.Tensor:
        value = global_probability[mode]
        if config.add_local_evidence:
            return value + peak_evidence(maps[mode], config)
        return value

    return {mode: (score(mode).float(), maps[mode].float())
            for mode in config.prompt_modes if mode in texts}


def training_logits(config: BackboneEvalConfig, backbone: Backbone, prompts,
                    images_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Image logits [B, 2] and per-layer pixel logits [L, B, 2, h, w].

    Pixel logits are returned per layer, not averaged: AnomalyCLIP applies the
    focal and dice terms to each layer's map and sums them, which supervises
    every layer that is read instead of letting one compensate for another.
    """
    with torch.autocast("cuda", enabled=(config.amp and config.device == "cuda")):
        features = backbone.encode(backbone.preprocess(images_uint8))
    text = prompts()
    logit_scale = logit_scale_for(config, backbone)
    return (global_logits(features, text, logit_scale, config,
                          backbone.has_two_global_tokens),
            dense_logits_per_layer(features["dense"], text, logit_scale))
