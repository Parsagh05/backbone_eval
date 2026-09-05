"""Exact text protocols for frozen and shallow learnable prompts."""

from __future__ import annotations

from bbeval.config import BackboneEvalConfig
from bbeval.datasets import prompt_class_name
from bbeval.prompts import (
    ANOMALOUS_STATES,
    FIXED_TEMPLATES,
    NORMAL_STATES,
    fixed_prompt_texts,
)


WINCLIP_TEMPLATES = (
    "a cropped photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of the {}.",
    "a dark photo of a {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.",
    "a photo of a {} for anomaly detection.",
)
WINCLIP_NORMAL_STATES = (
    "{}", "flawless {}", "perfect {}", "unblemished {}",
    "{} without flaw", "{} without defect", "{} without damage",
)
WINCLIP_ANOMALOUS_STATES = (
    "damaged {}", "{} with flaw", "{} with defect", "{} with damage",
)


def make_config(**overrides) -> BackboneEvalConfig:
    values = {"mvtec_root": "m", "visa_root": "v", "output_root": "o"}
    values.update(overrides)
    return BackboneEvalConfig(**values)


def test_frozen_vocabulary_is_the_published_winclip_ensemble():
    assert FIXED_TEMPLATES == WINCLIP_TEMPLATES
    assert NORMAL_STATES == WINCLIP_NORMAL_STATES
    assert ANOMALOUS_STATES == WINCLIP_ANOMALOUS_STATES


def test_frozen_prompts_insert_the_real_category_name():
    normal, anomalous = fixed_prompt_texts(make_config(), "metal_nut")
    assert len(normal) == 7 * 22 == 154
    assert len(anomalous) == 4 * 22 == 88
    assert normal[0] == "a cropped photo of the metal nut."
    assert normal[-1] == "a photo of a metal nut without damage for anomaly detection."
    assert anomalous[0] == "a cropped photo of the damaged metal nut."
    assert anomalous[-1] == "a photo of a metal nut with damage for anomaly detection."
    assert all("object" not in prompt for prompt in normal + anomalous)


def test_dataset_identifiers_are_not_collapsed_into_generic_names():
    assert prompt_class_name("metal_nut") == "metal nut"
    assert prompt_class_name("pcb1") == "pcb1"
    assert prompt_class_name("macaroni2") == "macaroni2"


def test_frozen_object_override_remains_an_explicit_ablation():
    config = make_config(fixed_prompt_class_name="object")
    normal, anomalous = fixed_prompt_texts(config, "bottle")
    assert normal[0] == "a cropped photo of the object."
    assert anomalous[0] == "a cropped photo of the damaged object."


def test_learnable_defaults_match_the_shallow_reference_repository():
    config = make_config()
    assert config.n_ctx == 12
    assert config.learnable_suffix == {
        "normal": "object.",
        "anomalous": "damaged object.",
    }
    assert config.init_std == 0.02
    assert config.map_temperature == 0.07
    assert config.grad_clip is None


def test_agnostic_variant_never_mentions_the_category():
    """`fixed_agnostic` labels every category "object", as the reference does."""
    from bbeval.prompts import AGNOSTIC_CLASS_NAME

    config = make_config()
    for category in ("hazelnut", "metal_nut", "pcb1"):
        normal, anomalous = fixed_prompt_texts(config, category,
                                               class_name=AGNOSTIC_CLASS_NAME)
        joined = " ".join(normal + anomalous)
        assert prompt_class_name(category) not in joined or category == "object"
        assert "object" in joined
    # Category-independent, which is why the sweep builds it once per backbone.
    assert (fixed_prompt_texts(config, "hazelnut", class_name=AGNOSTIC_CLASS_NAME)
            == fixed_prompt_texts(config, "pcb1", class_name=AGNOSTIC_CLASS_NAME))


def test_agnostic_and_class_aware_variants_differ():
    from bbeval.prompts import AGNOSTIC_CLASS_NAME

    config = make_config()
    aware = fixed_prompt_texts(config, "hazelnut")
    agnostic = fixed_prompt_texts(config, "hazelnut", class_name=AGNOSTIC_CLASS_NAME)
    assert aware != agnostic
    assert aware[0][0] == "a cropped photo of the hazelnut."
    assert agnostic[0][0] == "a cropped photo of the object."
    # Same ensemble size either way: only the label changes.
    assert [len(x) for x in aware] == [len(x) for x in agnostic]


def test_compact_ensemble_is_the_pre_winclip_vocabulary():
    """`fixed_compact` preserves the list used up to run cae0b9678540.

    Keeping it lets the jump that followed be attributed to the ensemble rather
    than to the other changes that landed in the same commit.
    """
    from bbeval.prompts import COMPACT_ENSEMBLE

    assert COMPACT_ENSEMBLE.templates == (
        "a photo of a {}.",
        "a cropped photo of a {}.",
        "a close-up photo of a {}.",
        "a bright photo of a {}.",
        "a dark photo of a {}.",
        "a blurry photo of a {}.",
        "a photo of a {} for visual inspection.",
    )
    assert COMPACT_ENSEMBLE.normal == (
        "{}", "flawless {}", "perfect {}", "unblemished {}",
        "{} without defect", "{} without damage")
    assert COMPACT_ENSEMBLE.anomalous == (
        "damaged {}", "flawed {}", "{} with defect",
        "{} with damage", "{} with flaw", "broken {}")


def test_frozen_modes_differ_only_where_intended():
    from bbeval.prompts import FIXED_MODES

    config = make_config()
    sizes = {mode: [len(part) for part in fixed_prompt_texts(config, "hazelnut", mode)]
             for mode in FIXED_MODES}
    # WinCLIP: 22 templates x 7 normal / 4 anomalous. Compact: 7 x 6 / 6.
    assert sizes["fixed"] == [154, 88]
    assert sizes["fixed_agnostic"] == [154, 88]
    assert sizes["fixed_compact"] == [42, 42]
    # fixed vs fixed_agnostic: same ensemble, different label.
    assert sizes["fixed"] == sizes["fixed_agnostic"]
    # fixed vs fixed_compact: same label, different ensemble.
    assert (fixed_prompt_texts(config, "hazelnut", "fixed")[0][0]
            .endswith("hazelnut."))
    assert (fixed_prompt_texts(config, "hazelnut", "fixed_compact")[0][0]
            .endswith("hazelnut."))
