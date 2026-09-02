"""Correctness gates for the SigLIP2 backbone.

Weights are random: every check here is a forward-path equivalence, which does
not depend on the weight values, so the suite needs no checkpoint download and
runs on CPU. Anything weight-dependent belongs on the GPU.

Evidence for each defect these gates guard against is in docs/siglip2_defects.md.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
open_clip = pytest.importorskip("open_clip")

import torch.nn.functional as F  # noqa: E402

from bbeval.backbones.siglip2 import SigLip2Backbone  # noqa: E402
from bbeval.config import BackboneEvalConfig  # noqa: E402

ARCH = "ViT-B-16-SigLIP2"
MODEL_ID = f"hf-hub:timm/{ARCH}"
TEXTS = ["a cropped photo of a flawless hazelnut",
         "a cropped photo of a damaged hazelnut",
         "a photo of a capsule with defect"]


@pytest.fixture(scope="module")
def shared_model():
    model = open_clip.create_model(ARCH, pretrained=None).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_config(tmp_path_factory, readout="map_token") -> BackboneEvalConfig:
    root = tmp_path_factory.mktemp("bbeval")
    return BackboneEvalConfig(
        mvtec_root=str(root / "mvtec"), visa_root=str(root / "visa"),
        output_root=str(root / "out"), device="cpu", n_ctx=8,
        siglip2_model=MODEL_ID, siglip2_dense_readout=readout,
        dense_layer_fractions={"siglip2": (1.0,)})


def build_backbone(monkeypatch, shared_model, config) -> SigLip2Backbone:
    monkeypatch.setattr(open_clip, "create_model_from_pretrained",
                        lambda *a, **k: (shared_model, None))
    return SigLip2Backbone(config)


@pytest.fixture(scope="module")
def backbone(shared_model, tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    config = make_config(tmp_path_factory)
    try:
        yield build_backbone(monkeypatch, shared_model, config)
    finally:
        monkeypatch.undo()


# --- D3: native resolution ---------------------------------------------------
def test_uses_native_resolution_not_shared_input_size(backbone):
    assert backbone.image_size == 224      # not the shared input_size of 518
    assert backbone.grid == 14
    assert backbone.image_size % backbone.patch_size == 0


# --- D4: logit_bias ----------------------------------------------------------
def test_logit_bias_is_retained(backbone):
    assert hasattr(backbone, "logit_bias")
    assert backbone.describe()["logit_bias"] == pytest.approx(backbone.logit_bias)


# --- D1 + D6: text forward ---------------------------------------------------
def test_text_forward_matches_open_clip_exactly(backbone, shared_model):
    with torch.no_grad():
        mine = backbone.encode_fixed_text(TEXTS)
        reference = shared_model.encode_text(backbone.tokenizer(TEXTS))
    assert (mine - reference).abs().max().item() == 0.0


def test_clip_style_argmax_pooling_would_be_wrong(backbone, shared_model):
    """The old path is not a small misalignment -- it is near-orthogonal."""
    tokenized = backbone.tokenizer(TEXTS)
    with torch.no_grad():
        embeds = backbone._tok_emb(tokenized)
        x = embeds + backbone._pos_emb.to(embeds.dtype)
        x = x.permute(1, 0, 2)                            # the batch-first defect
        x = backbone._text_tower(x)
        x = x.permute(1, 0, 2)
        x = backbone._ln_final(x)
        eot = tokenized.argmax(dim=-1)                    # the pooling defect
        old = backbone._text_proj(x[torch.arange(x.shape[0]), eot])
        reference = shared_model.encode_text(tokenized)
    # SentencePiece gives rare/capitalised pieces high ids, so the largest id in
    # a row is almost always the first token.
    assert (eot == 0).all()
    cosine = F.cosine_similarity(F.normalize(old, dim=-1),
                                 F.normalize(reference, dim=-1))
    assert cosine.abs().max().item() < 0.5


# --- D5: prompt slot alignment ----------------------------------------------
def test_learnable_slots_align_with_the_placeholder(backbone):
    ctx, aux = backbone.init_prompt(["object", "damaged object"])
    assert ctx.shape[:2] == (2, 8)
    assert ctx.shape[1] + aux["tail"].shape[1] == 64
    with torch.no_grad():
        assert backbone.encode_text(ctx, aux).shape == (2, backbone.embed_dim)


def test_fixed_suffix_survives_in_the_tail(backbone):
    """Rebuilding with the original placeholder must reproduce the plain text."""
    _, aux = backbone.init_prompt(["object", "damaged object"])
    with torch.no_grad():
        ids = backbone.tokenizer(["X " * 8 + "object", "X " * 8 + "damaged object"])
        original = backbone._tok_emb(ids)
        rebuilt = backbone._forward_text(
            torch.cat([original[:, :8], aux["tail"]], dim=1))
        plain = backbone._forward_text(original)
    assert (rebuilt - plain).abs().max().item() == 0.0


def test_placeholder_length_is_measured(backbone):
    assert backbone._placeholder_length("X " * 8) == 8


# --- D2: patch tokens reach the joint space ----------------------------------
def test_attention_pooling_over_all_tokens_reproduces_encode_image(backbone, shared_model):
    image = torch.randn(2, 3, backbone.image_size, backbone.image_size)
    trunk = backbone.visual.trunk
    with torch.no_grad():
        manual = trunk.attn_pool(trunk.forward_features(image))
        reference = shared_model.encode_image(image)
    assert (manual - reference).abs().max().item() == 0.0


def test_dense_tokens_have_joint_space_shape(backbone):
    image = torch.randn(2, 3, backbone.image_size, backbone.image_size)
    with torch.no_grad():
        dense = backbone.encode(image)["dense"]
    layer = sorted(dense)[-1]
    assert dense[layer].shape == (2, backbone.grid, backbone.grid, backbone.embed_dim)


def test_raw_control_differs_from_projected(shared_model, tmp_path_factory):
    """'raw' is the CLIP-shaped control; identical shape is why the bug was silent."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        projected = build_backbone(monkeypatch, shared_model,
                                   make_config(tmp_path_factory, "map_token"))
        raw = build_backbone(monkeypatch, shared_model,
                             make_config(tmp_path_factory, "raw"))
        image = torch.randn(2, 3, projected.image_size, projected.image_size)
        with torch.no_grad():
            a = projected.encode(image)["dense"]
            b = raw.encode(image)["dense"]
    finally:
        monkeypatch.undo()
    layer = sorted(a)[-1]
    assert a[layer].shape == b[layer].shape
    assert (a[layer] - b[layer]).abs().max().item() > 1e-4
