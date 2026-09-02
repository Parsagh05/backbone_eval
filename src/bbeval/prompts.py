"""Fixed prompt ensembles and the learnable context vectors.

`LearnablePrompts` holds the only trainable tensor anywhere in this package:
one parameter of shape [2, n_ctx, D]. The backbone is kept in a plain list so
it is not registered as a submodule -- the saved checkpoint then contains the
context vectors and nothing else, which is the audit trail for the claim that
no encoder parameter is touched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import Backbone
from .config import BackboneEvalConfig
from .datasets import prompt_class_name

FIXED_TEMPLATES = (
    "a photo of a {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of a {} for visual inspection.",
)
NORMAL_STATES = ("{}", "flawless {}", "perfect {}", "unblemished {}",
                 "{} without defect", "{} without damage")
ANOMALOUS_STATES = ("damaged {}", "flawed {}", "{} with defect",
                    "{} with damage", "{} with flaw", "broken {}")


def fixed_prompt_texts(config: BackboneEvalConfig,
                       category: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    name = config.fixed_prompt_class_name or prompt_class_name(category)
    return tuple(
        tuple(template.format(state.format(name))
              for template in FIXED_TEMPLATES for state in states)
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
    """The only trainable parameters: 2 x n_ctx context vectors."""

    def __init__(self, config: BackboneEvalConfig, backbone: Backbone) -> None:
        super().__init__()
        suffixes = [config.learnable_suffix["normal"],
                    config.learnable_suffix["anomalous"]]
        ctx, aux = backbone.init_prompt(suffixes)
        self.ctx = nn.Parameter(ctx.detach().float().clone())
        self._backbone = [backbone]                      # hidden from state_dict
        self._aux = aux

    def forward(self) -> torch.Tensor:
        embeddings = self._backbone[0].encode_text(self.ctx, self._aux)
        return F.normalize(embeddings.float(), dim=-1)   # [2, D]
