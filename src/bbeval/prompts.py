"""Fixed prompt ensembles and the learnable context vectors.

`LearnablePrompts` holds the only trainable tensors anywhere in this package:
one normal and one abnormal shallow context, each shaped [n_ctx, D]. The
backbone is kept in a plain list so it is not registered as a submodule -- the
saved checkpoint then contains the two context vectors and nothing else, which
is the audit trail for the claim that no encoder parameter is touched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import Backbone
from .config import BackboneEvalConfig
from .datasets import prompt_class_name

FIXED_TEMPLATES = (
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
NORMAL_STATES = ("{}", "flawless {}", "perfect {}", "unblemished {}",
                 "{} without flaw", "{} without defect", "{} without damage")
ANOMALOUS_STATES = ("damaged {}", "{} with flaw", "{} with defect",
                    "{} with damage")


def fixed_prompt_texts(config: BackboneEvalConfig,
                       category: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    name = config.fixed_prompt_class_name or prompt_class_name(category)
    return tuple(
        tuple(template.format(state.format(name))
              for state in states for template in FIXED_TEMPLATES)
        for states in (NORMAL_STATES, ANOMALOUS_STATES))


@torch.no_grad()
def build_fixed_text(config: BackboneEvalConfig, backbone: Backbone,
                     category: str) -> torch.Tensor:
    """Two prototypes: the mean unit embedding of each state's prompt ensemble."""
    prototypes = []
    for texts in fixed_prompt_texts(config, category):
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
