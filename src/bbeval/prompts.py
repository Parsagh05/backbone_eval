"""Fixed prompt ensembles and the learnable context vectors.

`LearnablePrompts` holds the only trainable tensors anywhere in this package:
one normal and one abnormal shallow context, each shaped [n_ctx, D]. The
backbone is kept in a plain list so it is not registered as a submodule -- the
saved checkpoint then contains the two context vectors and nothing else, which
is the audit trail for the claim that no encoder parameter is touched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import Backbone
from .config import FROZEN_PROMPT_MODES, BackboneEvalConfig
from .datasets import prompt_class_name

class PromptEnsemble(NamedTuple):
    """One frozen vocabulary: templates crossed with per-state phrasings."""

    templates: tuple[str, ...]
    normal: tuple[str, ...]
    anomalous: tuple[str, ...]

    def texts(self, name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return tuple(
            tuple(template.format(state.format(name))
                  for state in states for template in self.templates)
            for states in (self.normal, self.anomalous))


# WinCLIP's published compositional ensemble, verbatim and in order.
WINCLIP_ENSEMBLE = PromptEnsemble(
    templates=(
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
    ),
    normal=("{}", "flawless {}", "perfect {}", "unblemished {}",
            "{} without flaw", "{} without defect", "{} without damage"),
    anomalous=("damaged {}", "{} with flaw", "{} with defect", "{} with damage"),
)

# The smaller ensemble used up to run `cae0b9678540`. Retained so the jump that
# followed -- CLIP +8.7 pixel AUROC, SigLIP2 +0.8 -- can be attributed to the
# ensemble rather than to the other changes that landed in the same commit.
COMPACT_ENSEMBLE = PromptEnsemble(
    templates=(
        "a photo of a {}.",
        "a cropped photo of a {}.",
        "a close-up photo of a {}.",
        "a bright photo of a {}.",
        "a dark photo of a {}.",
        "a blurry photo of a {}.",
        "a photo of a {} for visual inspection.",
    ),
    normal=("{}", "flawless {}", "perfect {}", "unblemished {}",
            "{} without defect", "{} without damage"),
    anomalous=("damaged {}", "flawed {}", "{} with defect",
               "{} with damage", "{} with flaw", "broken {}"),
)

# The label used by the class-agnostic variant, matching the reference setup's
# CLASSNAME. Nothing about the category reaches the prompt.
AGNOSTIC_CLASS_NAME = "object"

# Category labels used alongside the compact ensemble up to run `cae0b9678540`,
# where several VisA identifiers were replaced by the object they name. Kept
# with that ensemble so `fixed_compact` reproduces the run exactly rather than
# differing on six VisA categories.
COMPACT_CLASS_NAMES = {
    "metal_nut": "metal nut", "pipe_fryum": "pipe fryum",
    "pcb1": "printed circuit board", "pcb2": "printed circuit board",
    "pcb3": "printed circuit board", "pcb4": "printed circuit board",
    "macaroni1": "macaroni", "macaroni2": "macaroni",
}


class FixedMode(NamedTuple):
    """One frozen prompt mode: a vocabulary plus how the label is chosen."""

    ensemble: PromptEnsemble
    label: Callable[[str], str]
    # False when the label ignores the category, so the sweep can encode the
    # ensemble once per backbone instead of once per category.
    category_dependent: bool


FIXED_MODES: dict[str, FixedMode] = {
    "fixed": FixedMode(WINCLIP_ENSEMBLE, prompt_class_name, True),
    "fixed_agnostic": FixedMode(WINCLIP_ENSEMBLE,
                                lambda _category: AGNOSTIC_CLASS_NAME, False),
    "fixed_compact": FixedMode(
        COMPACT_ENSEMBLE,
        lambda category: COMPACT_CLASS_NAMES.get(category,
                                                 prompt_class_name(category)),
        True),
}

assert tuple(FIXED_MODES) == FROZEN_PROMPT_MODES, (
    "config.FROZEN_PROMPT_MODES must list exactly the frozen modes defined here")

# Kept as names: tests/test_prompts.py pins this vocabulary against WinCLIP.
FIXED_TEMPLATES = WINCLIP_ENSEMBLE.templates
NORMAL_STATES = WINCLIP_ENSEMBLE.normal
ANOMALOUS_STATES = WINCLIP_ENSEMBLE.anomalous


def fixed_prompt_texts(config: BackboneEvalConfig, category: str,
                       mode: str = "fixed", class_name: str | None = None,
                       ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The frozen ensemble for one category under one frozen prompt mode.

    `class_name` overrides the label that fills every template, whatever the
    mode would otherwise use.
    """
    spec = FIXED_MODES[mode]
    name = class_name or config.fixed_prompt_class_name or spec.label(category)
    return spec.ensemble.texts(name)


@torch.no_grad()
def build_fixed_text(config: BackboneEvalConfig, backbone: Backbone,
                     category: str, mode: str = "fixed",
                     class_name: str | None = None) -> torch.Tensor:
    """Two prototypes: the mean unit embedding of each state's prompt ensemble."""
    prototypes = []
    for texts in fixed_prompt_texts(config, category, mode, class_name):
        embeddings = F.normalize(backbone.encode_fixed_text(list(texts)).float(), dim=-1)
        prototypes.append(F.normalize(embeddings.mean(dim=0), dim=-1))
    return torch.stack(prototypes)                       # [2, D]


class LearnablePrompts(nn.Module):
    """The only trainable parameters: normal and abnormal shallow contexts."""

    def __init__(self, config: BackboneEvalConfig, backbone: Backbone) -> None:
        super().__init__()
        suffixes = [config.learnable_suffix["normal"],
                    config.learnable_suffix["anomalous"]]
        ctx, aux = backbone.init_prompt(suffixes)
        self.normal_context = nn.Parameter(ctx[0].detach().float().clone())
        self.abnormal_context = nn.Parameter(ctx[1].detach().float().clone())
        self._backbone = [backbone]                      # hidden from state_dict
        self._aux = aux

    def context_tensor(self) -> torch.Tensor:
        return torch.stack((self.normal_context, self.abnormal_context), dim=0)

    def forward(self) -> torch.Tensor:
        embeddings = self._backbone[0].encode_text(self.context_tensor(), self._aux)
        return F.normalize(embeddings.float(), dim=-1)   # [2, D]
