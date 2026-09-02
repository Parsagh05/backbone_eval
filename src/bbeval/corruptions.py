"""Deterministic natural corruptions (pptx slides 18/19).

`imagecorruptions` is an optional dependency: a clean-only run must not require
it, so the import is deferred until a corruption is actually requested.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import BackboneEvalConfig
from .determinism import derived_seed

CLEAN = ("clean", 0)


def geometric_names(config: BackboneEvalConfig) -> tuple[str, ...]:
    return config.corruption_groups.get("geometric", ())


def selected_corruptions(config: BackboneEvalConfig) -> tuple[str, ...]:
    return tuple(name for group in config.corruption_groups.values() for name in group)


def corruption_group_of(config: BackboneEvalConfig) -> dict[str, str]:
    return {name: group
            for group, names in config.corruption_groups.items()
            for name in names}


def corruption_grid(config: BackboneEvalConfig) -> list[tuple[str, int]]:
    grid = [CLEAN] if config.include_clean else []
    if not config.corruptions_enabled:
        return grid or [CLEAN]
    return grid + [(name, severity)
                   for name in selected_corruptions(config)
                   for severity in config.severities]


def _imagenet_c_corrupt():
    try:
        from imagecorruptions import corrupt
    except Exception as error:  # noqa: BLE001
        raise ImportError(
            "imagecorruptions is required for photometric/noise/blur corruptions; "
            "install the 'corruptions' extra"
        ) from error
    return corrupt


def _affine_matrix(config: BackboneEvalConfig, corruption: str, severity: int, rng):
    """The 2x3 matrix mapping output normalised coordinates to input ones."""
    magnitude = config.geometric_magnitudes[corruption][severity - 1]
    if corruption == "rotation":
        angle = math.radians(magnitude) * rng.choice([-1.0, 1.0])
        cos, sin = math.cos(angle), math.sin(angle)
        return [[cos, -sin, 0.0], [sin, cos, 0.0]]
    if corruption == "zoom_scale":
        inverse = 1.0 / magnitude
        return [[inverse, 0.0, 0.0], [0.0, inverse, 0.0]]
    if corruption == "shift":
        # Normalised coordinates span [-1, 1], so a shift of `magnitude` side
        # lengths is 2 * magnitude in grid units.
        dx, dy = (2.0 * magnitude * rng.choice([-1.0, 1.0]) for _ in range(2))
        return [[1.0, 0.0, dx], [0.0, 1.0, dy]]
    raise ValueError(f"{corruption} is not a geometric perturbation")


def _warp(config: BackboneEvalConfig, image_uint8, mask_uint8, corruption, severity, seed):
    """Warps image and mask by one shared affine transform, replicating edges."""
    rng = np.random.default_rng(seed)
    theta = torch.tensor([_affine_matrix(config, corruption, severity, rng)],
                         dtype=torch.float32)
    # PIL hands over a read-only buffer, which torch refuses to wrap.
    image = torch.from_numpy(np.array(image_uint8, copy=True))
    image = image.permute(2, 0, 1)[None].float()
    grid = F.affine_grid(theta, list(image.shape), align_corners=False)
    warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="border",
                           align_corners=False)
    warped_image = (warped[0].permute(1, 2, 0)
                    .round().clamp(0, 255).to(torch.uint8).numpy())

    if mask_uint8 is None or not mask_uint8.any():
        return warped_image, mask_uint8
    mask = torch.from_numpy(np.array(mask_uint8, copy=True))[None, None].float()
    warped_mask = F.grid_sample(mask, grid, mode="nearest",
                                padding_mode="zeros", align_corners=False)
    return warped_image, warped_mask[0, 0].to(torch.uint8).numpy()


def apply_corruption(config: BackboneEvalConfig, image_uint8, mask_uint8,
                     corruption: str, severity: int, cache_key: str):
    """Corrupts an HxWx3 uint8 image, and its mask when the warp moves the defect.

    Reproducible by construction: the RNG is seeded from `cache_key` rather than
    drawn from a global stream, so an image gets the same corruption wherever in
    the sweep it is reached and a resumed run matches an uninterrupted one.
    """
    if corruption == "clean" or severity == 0:
        return image_uint8, mask_uint8

    seed = derived_seed(config.seed, cache_key, corruption, severity)
    if corruption in geometric_names(config):
        return _warp(config, image_uint8, mask_uint8, corruption, severity, seed)

    # A mask is meaningless for a photometric or noise corruption, which move no
    # pixels, so only the image changes.
    corrupt = _imagenet_c_corrupt()
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        corrupted = corrupt(image_uint8, corruption_name=corruption, severity=severity)
    finally:
        np.random.set_state(state)
    return corrupted, mask_uint8
