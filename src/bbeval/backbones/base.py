"""The frozen-encoder interface every backbone implements, plus the registry.

The contract is deliberately small. A backbone owns loading, preprocessing,
tokenisation, pooling and the map from its own tokens into the joint
image-text space; the rest of the pipeline owns prompts, scoring and metrics.

That split is what makes a comparison across backbones fair: each model
receives the same question in its own architecture's terms, rather than being
forced through another model's readout. Anything that differs per backbone is
therefore visible here and recorded by `describe()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import torch


class Backbone(ABC):
    """Frozen vision-language encoder with shallow prompt support."""

    name: str
    # True when the model exposes distinct "object" and "spatial" global tokens.
    has_two_global_tokens: bool = False

    # Populated by subclasses during __init__.
    embed_dim: int
    depth: int
    layers: tuple[int, ...]
    temperature: float
    image_size: int
    patch_size: int
    grid: int
    num_params: int

    @abstractmethod
    def preprocess(self, images_uint8: torch.Tensor) -> torch.Tensor:
        """uint8 [B,3,H,W] in [0,255] -> normalised float input for this model."""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> dict[str, Any]:
        """Return {"object": [B,D], "spatial": [B,D], "dense": {layer: [B,h,w,D]}}.

        `dense` tokens must already live in the joint image-text space, so that
        a cosine against a text embedding is meaningful.
        """

    @abstractmethod
    def init_prompt(self, suffixes: Sequence[str]) -> tuple[torch.Tensor, dict]:
        """Seed context vectors [K,N_CTX,D] plus whatever `encode_text` needs."""

    @abstractmethod
    def encode_text(self, ctx: torch.Tensor, aux: dict) -> torch.Tensor:
        """Encode learnable prompts -> [K,D]."""

    @abstractmethod
    def encode_fixed_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode plain strings -> [N,D]."""

    def describe(self) -> dict[str, Any]:
        """Per-backbone settings recorded with results, so choices stay auditable."""
        return {
            "backbone": self.name,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "dense_layers": list(self.layers),
            "temperature": self.temperature,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "grid": self.grid,
            "num_params": self.num_params,
        }


_BACKBONES: dict[str, Callable[..., Backbone]] = {}
_BACKBONE_ERRORS: dict[str, str] = {}
_T = TypeVar("_T", bound=Callable[..., Backbone])


def register_backbone(name: str) -> Callable[[_T], _T]:
    key = name.strip().lower()
    if not key:
        raise ValueError("backbone name cannot be empty")

    def decorate(factory: _T) -> _T:
        _BACKBONES[key] = factory
        _BACKBONE_ERRORS.pop(key, None)
        return factory

    return decorate


def skip_backbone(name: str, reason: object) -> None:
    """Record why a backbone is unavailable instead of raising at import time."""
    _BACKBONE_ERRORS[name.strip().lower()] = str(reason)


def backbone_names() -> tuple[str, ...]:
    return tuple(sorted(_BACKBONES))


def backbone_errors() -> dict[str, str]:
    return dict(_BACKBONE_ERRORS)


def create_backbone(name: str, **kwargs: Any) -> Backbone:
    key = name.strip().lower()
    if key not in _BACKBONES:
        reason = _BACKBONE_ERRORS.get(key)
        detail = f" ({reason})" if reason else ""
        raise ValueError(
            f"unknown backbone {name!r}{detail}; available: {backbone_names()}")
    return _BACKBONES[key](**kwargs)


def resolve_dense_layers(config, name: str, depth: int) -> tuple[int, ...]:
    """Turn depth fractions into 1-based block indices for a tower of `depth`."""
    fractions = (config.shared_dense_layers if config.shared_dense_layers is not None
                 else config.dense_layer_fractions.get(name, (1.0,)))
    return tuple(sorted({min(depth, max(1, int(round(fraction * depth))))
                         for fraction in fractions}))
