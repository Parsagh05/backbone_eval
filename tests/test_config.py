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


def test_defaults_follow_anomalyclip_evaluation_protocol():
    config = make()
    assert config.map_res == config.input_size == 518
    assert config.gaussian_sigma == 4.0
    assert config.add_local_evidence is False
    assert config.dense_layer_fractions["clip"] == (0.25, 0.5, 0.75, 1.0)
    assert config.dense_layer_fractions["siglip2"] == (1.0,)
    assert config.pixel_loss_layers == "last"


def test_fingerprint_tracks_settings_that_change_results():
    base = make()
    assert base.fingerprint() == make().fingerprint()
    # The readout is the whole point of the comparison, so it must change the id.
    assert base.fingerprint() != make(siglip2_dense_readout="raw").fingerprint()
    assert base.fingerprint() != make(seed=222).fingerprint()
    # Prompt fitting changes the learned map. Missing these meant a re-run with
    # different training would silently resume on the previous run's shards.
    assert base.fingerprint() != make(epochs=2).fingerprint()
    assert base.fingerprint() != make(learning_rate=1e-4).fingerprint()
    assert base.fingerprint() != make(max_train_images_per_category=20).fingerprint()
    assert base.fingerprint() != make(
        fixed_prompt_class_name="object").fingerprint()
    assert base.fingerprint() != make(
        learnable_suffix={"normal": "item.",
                          "anomalous": "damaged item."}).fingerprint()
    assert base.fingerprint() != make(focal_smooth=1e-4).fingerprint()
    assert base.fingerprint() != make(grad_clip=1.0).fingerprint()
    assert base.fingerprint() != make(pixel_loss_layers="all").fingerprint()


def test_fingerprint_includes_the_code_revision(monkeypatch):
    """A behaviour change must invalidate earlier artefacts, not resume on them."""
    import bbeval.config as config_module

    before = make().fingerprint()
    monkeypatch.setattr(config_module, "__version__", "9.9.9")
    assert make().fingerprint() != before


def test_fingerprint_ignores_settings_that_do_not_change_results():
    assert make().fingerprint() == make(
        num_workers=8, resume=False, archive_results=False).fingerprint()


@pytest.mark.parametrize("overrides", [
    {"loss_mode": "hinge"},
    {"siglip2_dense_readout": "cls"},
    {"prompt_modes": ["fixed", "nonsense"]},
    {"prompt_modes": []},
    {"n_ctx": 0},
    {"map_res": 0},
    {"focal_smooth": 0.5},
    {"grad_clip": 0.0},
    {"pixel_loss_layers": "mean"},
])
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        make(**overrides)


def test_any_frozen_mode_satisfies_the_protocol(recwarn):
    """Slide 23 asks for frozen and learnable, not for one specific vocabulary."""
    for frozen in ("fixed", "fixed_agnostic", "fixed_compact"):
        config = make(prompt_modes=[frozen, "learned"])
        assert config.prompt_modes == (frozen, "learned")
    assert not recwarn.list


@pytest.mark.parametrize("modes, omitted", [
    (["fixed_compact"], "learned"),
    (["learned"], "frozen"),
])
def test_partial_protocol_warns_rather_than_failing(modes, omitted):
    """A frozen-only run needs no training, and shards compose across runs."""
    with pytest.warns(UserWarning, match=omitted):
        config = make(prompt_modes=modes)
    assert config.prompt_modes == tuple(modes)
