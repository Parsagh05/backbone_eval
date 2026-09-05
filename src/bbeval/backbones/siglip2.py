"""SigLIP2 backbone (OpenCLIP / timm).

SigLIP2 differs from CLIP in ways that a CLIP-shaped harness gets silently
wrong -- see docs/siglip2_defects.md for the evidence behind each choice here:

* the text embedding is pooled at the **last** sequence position, not at an
  argmax-over-token-ids EOT slot;
* OpenCLIP's SigLIP text tower runs **batch-first**, so the sequence must not
  be transposed;
* the tokenizer prepends **no BOS**, so learnable context occupies [0, n_ctx);
* there is no per-token projection: `timm_proj` is "none" and `timm_pool` is
  "map", so the joint space is defined solely by `trunk.attn_pool`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from ..config import BackboneEvalConfig
from .base import Backbone, register_backbone, resolve_dense_layers, skip_backbone

try:
    import open_clip
except Exception as _open_clip_error:  # noqa: BLE001
    open_clip = None
    skip_backbone("siglip2", _open_clip_error)
    skip_backbone("siglip2_grid37", _open_clip_error)

SIGLIP2_MEAN = (0.5, 0.5, 0.5)
SIGLIP2_STD = (0.5, 0.5, 0.5)


def _project_text(pooled: torch.Tensor, projection) -> torch.Tensor:
    if projection is None:
        return pooled
    if isinstance(projection, torch.nn.Linear):
        return projection(pooled)
    return pooled @ projection


def _text_modules(model):
    """The pieces of an OpenCLIP SigLIP text tower needed for prompt learning."""
    text = getattr(model, "text", None)
    if text is None or not hasattr(text, "transformer"):
        raise TypeError("SigLIP2 expects an OpenCLIP custom text tower (model.text)")
    projection = getattr(text, "text_projection", None) or getattr(text, "proj", None)
    return {
        "token_embedding": text.token_embedding,
        "positional_embedding": text.positional_embedding,
        "transformer": text.transformer,
        "ln_final": text.ln_final,
        "projection": projection,
        "attn_mask": getattr(text, "attn_mask", None),
        "pool_type": getattr(text, "pool_type", "last"),
    }


class SigLip2Backbone(Backbone):
    """Frozen SigLIP2 encoder with the shared prompt-learning interface."""

    name = "siglip2"
    has_two_global_tokens = False
    forced_patch_grid: int | None = None

    def __init__(self, config: BackboneEvalConfig) -> None:
        if open_clip is None:
            raise ImportError("open-clip-torch is required for the siglip2 backbone")
        self.config = config
        self.device = config.device
        self.model_id = config.siglip2_model
        self.dense_readout = config.siglip2_dense_readout

        cache_dir = os.path.join(config.weights_dir, "open_clip") if config.weights_dir else None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        loaded = (open_clip.create_model_from_pretrained(self.model_id, cache_dir=cache_dir)
                  if cache_dir else
                  open_clip.create_model_from_pretrained(self.model_id))
        model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        self.tokenizer = open_clip.get_tokenizer(self.model_id)
        self.model = model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.visual = self.model.visual
        self.patch_size = self._infer_patch_size()
        self.native_image_size = self._native_image_size()
        # The ordinary variant stays checkpoint-native. `siglip2_grid37`
        # intentionally uses 37 * patch_size (592 for L/16) and exercises the
        # explicit positional-interpolation path below.
        requested_size = (self.forced_patch_grid * self.patch_size
                          if self.forced_patch_grid is not None
                          else config.siglip2_input_size)
        self.image_size = int(requested_size or self.native_image_size)
        if self.image_size % self.patch_size:
            raise ValueError(
                f"SigLIP2 image size {self.image_size} is not a multiple of "
                f"patch size {self.patch_size}")
        self.grid = self.image_size // self.patch_size
        if self.image_size != self.native_image_size:
            patch_embed = self.visual.trunk.patch_embed
            if hasattr(patch_embed, "strict_img_size"):
                patch_embed.strict_img_size = False

        modules = _text_modules(self.model)
        self._tok_emb = modules["token_embedding"]
        self._pos_emb = modules["positional_embedding"]
        self._text_tower = modules["transformer"]
        self._ln_final = modules["ln_final"]
        self._text_proj = modules["projection"]
        self._attn_mask = modules["attn_mask"]
        # CLIP's argmax-over-token-ids trick does not transfer: on a 256k
        # SentencePiece vocab the largest id is an arbitrary word piece, and in
        # practice argmax lands on position 0 for almost every prompt.
        if modules["pool_type"] != "last":
            raise ValueError(
                f"SigLIP2 text tower reports pool_type={modules['pool_type']!r}; "
                "this backbone implements 'last' only")

        self.embed_dim = self._infer_embed_dim()
        self.depth = self._infer_depth()
        self.layers = resolve_dense_layers(config, self.name, self.depth)
        self.temperature = float(1.0 / self.model.logit_scale.exp().item())
        # Kept for sigmoid-native scoring and the calibration track. It cancels
        # in a two-class softmax, so it does not affect the default metrics.
        bias = getattr(self.model, "logit_bias", None)
        self.logit_bias = float(bias.item()) if bias is not None else 0.0
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self._mean = torch.tensor(SIGLIP2_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(SIGLIP2_STD, device=self.device).view(1, 3, 1, 1)

    # --- introspection -------------------------------------------------------
    def _infer_embed_dim(self) -> int:
        projection = self._text_proj
        if isinstance(projection, torch.nn.Linear):
            return int(projection.out_features)
        if isinstance(projection, torch.Tensor):
            return int(projection.shape[1])
        return int(getattr(self.model, "embed_dim", 1024))

    def _infer_patch_size(self) -> int:
        trunk = getattr(self.visual, "trunk", None)
        patch = getattr(getattr(trunk, "patch_embed", None), "patch_size", None)
        if patch is not None:
            return int(patch[0] if isinstance(patch, (tuple, list)) else patch)
        return 16

    def _native_image_size(self) -> int:
        trunk = getattr(self.visual, "trunk", None)
        size = getattr(getattr(trunk, "patch_embed", None), "img_size", None)
        if isinstance(size, (tuple, list)):
            return int(size[0])
        return int(size) if size else self.config.input_size

    def _infer_depth(self) -> int:
        trunk = getattr(self.visual, "trunk", None)
        if trunk is not None and hasattr(trunk, "blocks"):
            return len(trunk.blocks)
        return 24

    def describe(self) -> dict[str, Any]:
        return {**super().describe(),
                "model_id": self.model_id,
                "dense_readout": self.dense_readout,
                "native_image_size": self.native_image_size,
                "positional_embedding_interpolated": (
                    self.image_size != self.native_image_size),
                "logit_bias": self.logit_bias}

    # --- vision --------------------------------------------------------------
    def preprocess(self, images_uint8: torch.Tensor) -> torch.Tensor:
        x = images_uint8.to(self.device, non_blocking=True).float().div_(255.0)
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bicubic", align_corners=False, antialias=True)
        return (x.clamp_(0.0, 1.0) - self._mean) / self._std

    def _project_dense(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map trunk tokens into the joint image-text space.

        SigLIP2 has no CLS token and no per-token projection, so pooling a
        single token through `trunk.attn_pool` reuses exactly the function that
        produced the image embedding. Given the full token set this chain
        reproduces `encode_image` bit-for-bit (see tests).
        """
        trunk = self.visual.trunk
        normed = trunk.norm(tokens)
        if self.dense_readout == "raw":
            return normed
        batch, length, width = normed.shape
        pooled = trunk.attn_pool(normed.reshape(batch * length, 1, width))
        pooled = trunk.head(trunk.head_drop(trunk.fc_norm(pooled)))
        pooled = self.visual.head(pooled)
        return pooled.reshape(batch, length, -1)

    def _project_token_set(self, tokens: torch.Tensor) -> torch.Tensor:
        """The checkpoint's native full-set MAP head, without another trunk pass."""
        trunk = self.visual.trunk
        pooled = trunk.attn_pool(trunk.norm(tokens))
        pooled = trunk.head(trunk.head_drop(trunk.fc_norm(pooled)))
        return self.visual.head(pooled)

    def _interpolate_position_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add the checkpoint position grid after bicubic grid interpolation."""
        trunk = self.visual.trunk
        if int(getattr(trunk, "num_prefix_tokens", 0)) != 0:
            raise ValueError("SigLIP2 grid interpolation expects no prefix tokens")
        position = getattr(trunk, "pos_embed", None)
        if position is None or position.ndim != 3:
            raise ValueError("SigLIP2 grid interpolation needs a [1,N,D] pos_embed")

        source = int(round(position.shape[1] ** 0.5))
        target = int(round(tokens.shape[1] ** 0.5))
        if source * source != position.shape[1] or target * target != tokens.shape[1]:
            raise ValueError("SigLIP2 positional tokens must form square patch grids")
        if source != target:
            position = position.float().reshape(1, source, source, -1)
            position = position.permute(0, 3, 1, 2)
            position = F.interpolate(position, size=(target, target), mode="bicubic",
                                     align_corners=False, antialias=True)
            position = position.permute(0, 2, 3, 1).reshape(1, target * target, -1)
        return tokens + position.to(device=tokens.device, dtype=tokens.dtype)

    def encode(self, x: torch.Tensor) -> dict[str, Any]:
        trunk = self.visual.trunk
        tokens = trunk.patch_embed(x)
        off_native = self.image_size != self.native_image_size
        if off_native:
            tokens = self._interpolate_position_embedding(tokens)
            if hasattr(trunk, "pos_drop"):
                tokens = trunk.pos_drop(tokens)
        elif hasattr(trunk, "_pos_embed"):
            tokens = trunk._pos_embed(tokens)
        elif hasattr(trunk, "pos_embed"):
            tokens = tokens + trunk.pos_embed
        if not hasattr(trunk, "_pos_embed") and not off_native and hasattr(trunk, "pos_drop"):
            tokens = trunk.pos_drop(tokens)
        if hasattr(trunk, "patch_drop"):
            tokens = trunk.patch_drop(tokens)
        if hasattr(trunk, "norm_pre"):
            tokens = trunk.norm_pre(tokens)
        prefix = int(getattr(trunk, "num_prefix_tokens", 0))

        dense = {}
        for depth, block in enumerate(trunk.blocks, start=1):
            tokens = block(tokens)
            if depth in self.layers:
                dense[depth] = tokens[:, prefix:]

        # At native resolution, use the model's own complete image forward.
        # Off-native, its fixed positional grid cannot be called directly, so
        # apply the exact same frozen MAP/head chain to our interpolated tokens.
        global_embedding = (self._project_token_set(tokens) if off_native
                            else self.model.encode_image(x))
        if global_embedding.ndim > 2:
            global_embedding = global_embedding.mean(dim=1)

        projected = {layer: self._project_dense(patches)
                     for layer, patches in dense.items()}
        grid = (int(round(next(iter(dense.values())).shape[1] ** 0.5))
                if dense else self.grid)
        return {
            "object": global_embedding,
            "spatial": global_embedding,
            "dense": {layer: value.reshape(value.shape[0], grid, grid, -1)
                      for layer, value in projected.items()},
        }

    # --- text ----------------------------------------------------------------
    def _placeholder_length(self, placeholder: str) -> int:
        """Token count of the learnable placeholder, measured not assumed."""
        row = self.tokenizer([placeholder])[0].tolist()
        pad = row[-1]
        body = [token for token in row if token != pad]
        return max(len(body) - 1, 0)          # drop the trailing EOS

    def init_prompt(self, suffixes: Sequence[str]) -> tuple[torch.Tensor, dict]:
        n_ctx = self.config.n_ctx
        placeholder = "X " * n_ctx
        texts = [f"{placeholder}{suffix}" for suffix in suffixes]
        tokenized = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            embeds = self._tok_emb(tokenized)
        # SigLIP's SentencePiece tokenizer prepends no BOS, so the placeholder
        # occupies [0, n_ctx) -- not [1, 1+n_ctx) as in CLIP. Measure it: a
        # tokenizer change would otherwise slide the learnable slots into the
        # fixed suffix with no error at all.
        measured = self._placeholder_length(placeholder.strip())
        if measured != n_ctx:
            raise ValueError(
                f"SigLIP2 tokenizer maps the {n_ctx}-token placeholder to "
                f"{measured} tokens; learnable-prompt slicing would be misaligned")
        # AnomalyCLIP draws the context from N(0, init_std) rather than seeding
        # it from the "X" placeholder embeddings.
        ctx = torch.empty(len(texts), n_ctx, embeds.shape[-1],
                          dtype=embeds.dtype, device=embeds.device)
        torch.nn.init.normal_(ctx, std=self.config.init_std)
        return ctx, {"tail": embeds[:, n_ctx:].clone()}

    def _forward_text(self, embeds: torch.Tensor) -> torch.Tensor:
        # OpenCLIP's Transformer runs batch-first for SigLIP, so the sequence
        # must NOT be transposed into [L, B, D].
        x = embeds + self._pos_emb[:embeds.shape[1]].to(embeds.dtype)
        x = self._text_tower(x, attn_mask=self._attn_mask)
        x = self._ln_final(x)
        return _project_text(x[:, -1], self._text_proj)      # pool_type == "last"

    def encode_text(self, ctx: torch.Tensor, aux: dict) -> torch.Tensor:
        return self._forward_text(torch.cat([ctx, aux["tail"]], dim=1))

    @torch.no_grad()
    def encode_fixed_text(self, texts: Sequence[str]) -> torch.Tensor:
        tokenized = self.tokenizer(list(texts)).to(self.device)
        return self._forward_text(self._tok_emb(tokenized))


class SigLip2Grid37Backbone(SigLip2Backbone):
    """SigLIP2 control resized so patch-16 produces the shared 37x37 grid."""

    name = "siglip2_grid37"
    forced_patch_grid = 37


if open_clip is not None:
    register_backbone("siglip2")(SigLip2Backbone)
    register_backbone("siglip2_grid37")(SigLip2Grid37Backbone)
