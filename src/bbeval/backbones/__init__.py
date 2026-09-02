"""Backbone registry.

Importing the submodules registers each backbone, or records why it is
unavailable, so a missing optional dependency never breaks the whole run.
"""

from .base import (
    Backbone,
    backbone_errors,
    backbone_names,
    create_backbone,
    register_backbone,
    resolve_dense_layers,
    skip_backbone,
)
from . import clip as _clip  # noqa: F401
from . import siglip2 as _siglip2  # noqa: F401

__all__ = [
    "Backbone",
    "backbone_errors",
    "backbone_names",
    "create_backbone",
    "register_backbone",
    "resolve_dense_layers",
    "skip_backbone",
]
