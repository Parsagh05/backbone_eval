"""OpenAI CLIP backbone -- the reference against which other backbones are read.

CLIP's joint image-text space is `visual.proj @ ln_post(token)`: one linear map
applied identically to every token. That is why patch tokens can be compared to
text embeddings for free, and it is the assumption the rest of this literature
inherits. Backbones that pool differently must supply their own equivalent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from ..config import BackboneEvalConfig
from .base import Backbone, register_backbone, resolve_dense_layers, skip_backbone

try:
    import clip
except Exception as _clip_error:  # noqa: BLE001
    clip = None
    skip_backbone("clip", _clip_error)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _dpam_value_attention(block, x: torch.Tensor) -> torch.Tensor:
    """Official AnomalyCLIP DPAM V-V attention output.

    AnomalyCLIP copies a CLIP block's QKV/output weights into a dual-path
    attention module, then replaces Q and K with V for its dense branch. This
    functional form avoids mutating the frozen OpenAI CLIP model; the ordinary
    block still computes the untouched global path.
    """
    normed = block.ln_1(x)
    qkv = F.linear(normed, block.attn.in_proj_weight, block.attn.in_proj_bias)
    value = qkv.chunk(3, dim=-1)[2]
    length, batch, width = value.shape
    heads = block.attn.num_heads
    head_dim = width // heads
    scale = head_dim ** -0.5

    def split(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor.permute(1, 0, 2)
                .reshape(batch, length, heads, head_dim)
                .permute(0, 2, 1, 3))

    value = split(value)
    weights = ((value * scale) @ value.transpose(-2, -1)).softmax(dim=-1)
    output = (weights @ value).permute(0, 2, 1, 3).reshape(batch, length, width)
    return block.attn.out_proj(output.permute(1, 0, 2))


class ClipBackbone(Backbone):
    name = "clip"
    has_two_global_tokens = False

    def __init__(self, config: BackboneEvalConfig) -> None:
        if clip is None:
            raise ImportError("the openai/CLIP package is required for the clip backbone")
        self.config = config
        self.device = config.device
        model, _ = clip.load(config.clip_backbone, device=self.device, jit=False,
                             download_root=config.weights_dir or None)
        self.model = model.float().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.visual = self.model.visual
        self.patch_size = self.visual.conv1.kernel_size[0]
        self.image_size = config.input_size
        self.grid = self.image_size // self.patch_size
        self.embed_dim = self.model.text_projection.shape[1]
        self.depth = len(self.visual.transformer.resblocks)
        self.layers = resolve_dense_layers(config, self.name, self.depth)
        self.temperature = float(1.0 / self.model.logit_scale.exp().item())
        self.num_params = sum(p.numel() for p in self.model.parameters())

        self._pos_embed = self._interpolated_pos_embed()
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    def _interpolated_pos_embed(self) -> torch.Tensor:
        """Resize the visual positional embedding to the configured input size."""
        pos = self.visual.positional_embedding.detach()
        class_pos, patch_pos = pos[:1], pos[1:]
        native = int(round(patch_pos.shape[0] ** 0.5))
        if native == self.grid:
            return pos
        patch_pos = patch_pos.reshape(1, native, native, -1).permute(0, 3, 1, 2)
        # AnomalyCLIP's visual path uses bilinear positional interpolation.
        patch_pos = F.interpolate(patch_pos, size=(self.grid, self.grid),
                                  mode="bilinear", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(self.grid ** 2, -1)
        return torch.cat([class_pos, patch_pos], dim=0)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(),
                "model_id": self.config.clip_backbone,
                "dense_readout": ("anomalyclip_dpam" if self.config.use_value_attention
                                  else "raw_ln_post_proj"),
                "dpam_start_layer": (max(1, self.depth - 18)
                                     if self.config.use_value_attention else None),
                "value_attention": self.config.use_value_attention}

    # --- vision --------------------------------------------------------------
    def preprocess(self, images_uint8: torch.Tensor) -> torch.Tensor:
        x = images_uint8.to(self.device, non_blocking=True).float().div_(255.0)
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bicubic", align_corners=False, antialias=True)
        return (x.clamp_(0.0, 1.0) - self._mean) / self._std

    def _project(self, tokens: torch.Tensor) -> torch.Tensor:
        """Into the joint image-text space, where the prompt embeddings live."""
        return self.visual.ln_post(tokens) @ self.visual.proj

    def encode(self, x: torch.Tensor) -> dict[str, Any]:
        visual = self.visual
        grid = x.shape[-1] // self.patch_size

        tokens = visual.conv1(x)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
        class_token = (visual.class_embedding.to(tokens.dtype)
                       .view(1, 1, -1).expand(tokens.shape[0], 1, -1))
        tokens = torch.cat([class_token, tokens], dim=1) + self._pos_embed
        hidden = visual.ln_pre(tokens).permute(1, 0, 2)      # NLD -> LND

        dense = {}
        dpam = None
        # Official DAPM_replace(20) replaces the final 19 blocks: layer 6
        # onward for the 24-block ViT-L/14 used by AnomalyCLIP.
        dpam_start = max(1, self.depth - 18)
        for depth, block in enumerate(visual.transformer.resblocks, start=1):
            previous = hidden
            hidden = block(hidden)
            if self.config.use_value_attention and depth >= dpam_start:
                value_attention = _dpam_value_attention(block, previous)
                # Official ViT-L DPAM starts a second residual stream at its
                # first selected stage (layer 6), accumulates V-V attention,
                # and deliberately skips the dense branch's FFNs.
                dpam = ((previous if dpam is None else dpam) + value_attention)
                source = dpam
            else:
                source = hidden
            if depth in self.layers:
                dense[depth] = self._project(source.permute(1, 0, 2)[:, 1:])
        global_embedding = self._project(hidden.permute(1, 0, 2)[:, :1])[:, 0]

        return {
            "object": global_embedding,
            "spatial": global_embedding,          # CLIP has a single global token
            "dense": {layer: value.reshape(value.shape[0], grid, grid, -1)
                      for layer, value in dense.items()},
        }

    # --- text ----------------------------------------------------------------
    def init_prompt(self, suffixes: Sequence[str]) -> tuple[torch.Tensor, dict]:
        """[SOT][V1..Vn][suffix][EOT], with only the V tokens trainable."""
        n_ctx = self.config.n_ctx
        texts = [f"{'X ' * n_ctx}{suffix}" for suffix in suffixes]
        tokenized = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            embeds = self.model.token_embedding(tokenized)
        # AnomalyCLIP draws the context from N(0, init_std) rather than seeding
        # it from the "X" placeholder embeddings: at that scale the fixed suffix
        # dominates the prompt's meaning early, so "object." and "damaged object."
        # start apart instead of nearly on top of each other.
        ctx = torch.empty(len(texts), n_ctx, embeds.shape[-1],
                          dtype=embeds.dtype, device=embeds.device)
        torch.nn.init.normal_(ctx, std=self.config.init_std)
        aux = {
            "prefix": embeds[:, :1].clone(),               # start-of-text
            "tail": embeds[:, 1 + n_ctx:].clone(),         # suffix, EOT, padding
            "eot_index": tokenized.argmax(dim=-1),
        }
        return ctx, aux

    def _forward_text(self, embeds: torch.Tensor, eot_index: torch.Tensor) -> torch.Tensor:
        model = self.model
        x = embeds + model.positional_embedding
        x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = model.ln_final(x)
        pooled = x[torch.arange(x.shape[0], device=x.device), eot_index]
        return pooled @ model.text_projection

    def encode_text(self, ctx: torch.Tensor, aux: dict) -> torch.Tensor:
        embeds = torch.cat([aux["prefix"], ctx, aux["tail"]], dim=1)
        return self._forward_text(embeds, aux["eot_index"])

    @torch.no_grad()
    def encode_fixed_text(self, texts: Sequence[str]) -> torch.Tensor:
        tokenized = clip.tokenize(list(texts)).to(self.device)
        return self._forward_text(self.model.token_embedding(tokenized),
                                  tokenized.argmax(dim=-1))


if clip is not None:
    register_backbone("clip")(ClipBackbone)
