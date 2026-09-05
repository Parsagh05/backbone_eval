"""Parity gates for AnomalyCLIP inference-time score construction."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bbeval.config import BackboneEvalConfig  # noqa: E402
from bbeval.scoring import to_anomaly_map  # noqa: E402


def config(**overrides) -> BackboneEvalConfig:
    payload = {
        "mvtec_root": "x",
        "visa_root": "y",
        "output_root": "z",
        "map_res": 2,
        "gaussian_sigma": 0.0,
    }
    payload.update(overrides)
    return BackboneEvalConfig(**payload)


def test_map_sums_per_layer_probabilities_instead_of_softmaxing_mean_logits():
    # Abnormal probabilities are 0.9 and 0.2, so the official per-layer sum is
    # 1.1 everywhere. This also pins that a four-layer map is not clamped to 1.
    logits = torch.tensor(
        [
            [[[[0.0]], [[2.1972246]]]],
            [[[[1.3862944]], [[0.0]]]],
        ]
    )
    anomaly_map = to_anomaly_map(logits, config())
    assert anomaly_map.shape == (1, 2, 2)
    assert torch.allclose(anomaly_map, torch.full_like(anomaly_map, 1.1), atol=1e-6)


def test_single_layer_logits_remain_supported_for_control_runs():
    logits = torch.tensor([[[[0.0]], [[0.0]]]])
    anomaly_map = to_anomaly_map(logits, config(map_res=3))
    assert anomaly_map.shape == (1, 3, 3)
    assert torch.equal(anomaly_map, torch.full_like(anomaly_map, 0.5))
