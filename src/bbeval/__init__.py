"""Backbone benchmarking for zero-shot anomaly detection."""

from ._version import __version__
from .config import BackboneEvalConfig
from .engine import run_evaluation

__all__ = ["BackboneEvalConfig", "run_evaluation", "__version__"]
