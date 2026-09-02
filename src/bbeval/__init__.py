"""Backbone benchmarking for zero-shot anomaly detection."""

from .config import BackboneEvalConfig
from .engine import run_evaluation

__all__ = ["BackboneEvalConfig", "run_evaluation"]
__version__ = "0.1.0"
