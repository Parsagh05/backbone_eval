"""CLIP dense-readout gates.

Needs real weights, so the suite skips unless a checkpoint is already cached.
Point `BBEVAL_CLIP_WEIGHTS` at a directory containing `ViT-B-32.pt`, or let it
find the default `~/.cache/clip`. Nothing here downloads anything.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("clip")

from bbeval.backbones.clip import ClipBackbone  # noqa: E402
from bbeval.config import BackboneEvalConfig  # noqa: E402

MODEL = "ViT-B/32"
CHECKPOINT = "ViT-B-32.pt"
LAYERS = (0.25, 0.5, 0.75, 1.0)


def weights_dir() -> str:
    candidates = [os.environ.get("BBEVAL_CLIP_WEIGHTS"),
                  str(Path.home() / ".cache" / "clip")]
    for candidate in candidates:
        if candidate and (Path(candidate) / CHECKPOINT).is_file():
            return candidate
    pytest.skip(f"no cached {CHECKPOINT}; set BBEVAL_CLIP_WEIGHTS to run")


@pytest.fixture(scope="module")
def backbones(tmp_path_factory):
    cache = weights_dir()

    def build(value_attention: bool) -> ClipBackbone:
        return ClipBackbone(BackboneEvalConfig(
            mvtec_root="x", visa_root="y",
            output_root=str(tmp_path_factory.mktemp("clip")),
            weights_dir=cache, device="cpu", clip_backbone=MODEL,
            input_size=224, n_ctx=8, amp=False,
            use_value_attention=value_attention,
            dense_layer_fractions={"clip": LAYERS}))

    return build(True), build(False)


def test_dpam_reaches_every_selected_layer(backbones):
    """Raw CLIP patch-text cosine is anti-correlated with the object.

    The DPAM side branch starts at the first selected stage and accumulates
    V-V attention through every later selected stage.
    """
    surgery, plain = backbones
    image = torch.randn(2, 3, surgery.image_size, surgery.image_size)
    with torch.no_grad():
        patched, raw = surgery.encode(image), plain.encode(image)

    assert len(surgery.layers) > 1, "this gate is meaningless with a single layer"
    for layer in surgery.layers:
        difference = (patched["dense"][layer] - raw["dense"][layer]).abs().max().item()
        assert difference > 1e-6, f"layer {layer} was left unpatched"


def test_surgery_leaves_the_global_embedding_untouched(backbones):
    """The side branch must not disturb the image-level score."""
    surgery, plain = backbones
    image = torch.randn(2, 3, surgery.image_size, surgery.image_size)
    with torch.no_grad():
        patched, raw = surgery.encode(image), plain.encode(image)
    assert (patched["object"] - raw["object"]).abs().max().item() == 0.0


def test_global_embedding_matches_clips_own_encode_image(backbones):
    surgery, _ = backbones
    image = torch.randn(2, 3, surgery.image_size, surgery.image_size)
    with torch.no_grad():
        mine = surgery.encode(surgery.preprocess(
            (image.clamp(-1, 1) * 127 + 128).to(torch.uint8)))["object"]
        reference = surgery.model.encode_image(surgery.preprocess(
            (image.clamp(-1, 1) * 127 + 128).to(torch.uint8)))
    assert (mine - reference).abs().max().item() == 0.0
