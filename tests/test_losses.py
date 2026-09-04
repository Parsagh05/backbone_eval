"""Numerical parity with the shallow AnomalyCLIP-style reference loss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from bbeval.losses import focal_loss  # noqa: E402


def test_focal_loss_matches_reference_smoothing_formula():
    probabilities = torch.tensor(
        [[[[0.8, 0.3]], [[0.2, 0.7]]]], dtype=torch.float32)
    target = torch.tensor([[[0, 1]]])
    gamma, smooth = 2.0, 1e-5

    one_hot = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
    one_hot = one_hot.clamp(smooth, 1.0 - smooth)
    truth = (one_hot * probabilities).sum(dim=1) + smooth
    expected = (-((1.0 - truth) ** gamma) * truth.log()).mean()

    assert torch.equal(focal_loss(probabilities, target, gamma, smooth), expected)
