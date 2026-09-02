"""Configuration validation. No torch, no weights, no data."""

from __future__ import annotations

import pytest

from bbeval.config import BackboneEvalConfig


def make(**overrides) -> BackboneEvalConfig:
    payload = {"mvtec_root": "/data/mvtec", "visa_root": "/data/visa",
               "output_root": "/out"}
    payload.update(overrides)
    return BackboneEvalConfig(**payload)


def test_sequences_are_normalised_to_tuples():
    config = make(backbones=["clip"], severities=[1, 2],
                  protocol=[["mvtec", "visa"]],
                  categories={"mvtec": ["hazelnut"]})
    assert config.backbones == ("clip",)
    assert config.severities == (1, 2)
    assert config.protocol == (("mvtec", "visa"),)
    assert config.categories == {"mvtec": ("hazelnut",)}


def test_derived_paths_hang_off_output_root():
    config = make(output_root="/out")
    assert config.artifact_dir == "/out/artifacts"
    assert config.checkpoint_dir == "/out/prompts"
    assert config.table_dir == "/out/tables"
    assert config.dataset_roots == {"mvtec": "/data/mvtec", "visa": "/data/visa"}


def test_fingerprint_tracks_settings_that_change_results():
    base = make()
    assert base.fingerprint() == make().fingerprint()
    # The readout is the whole point of the comparison, so it must change the id.
    assert base.fingerprint() != make(siglip2_dense_readout="raw").fingerprint()
    assert base.fingerprint() != make(seed=222).fingerprint()


def test_fingerprint_ignores_settings_that_do_not_change_results():
    assert make().fingerprint() == make(num_workers=8, resume=False).fingerprint()


@pytest.mark.parametrize("overrides", [
    {"loss_mode": "hinge"},
    {"siglip2_dense_readout": "cls"},
    {"prompt_modes": ["fixed", "nonsense"]},
    {"prompt_modes": ["fixed"]},          # slide 23 needs fixed AND learned
    {"n_ctx": 0},
    {"map_res": 0},
])
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        make(**overrides)
