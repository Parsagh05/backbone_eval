"""Dependency-free gates for the equal-patch-grid SigLIP2 control."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from bbeval.backbones.siglip2 import (  # noqa: E402
    SigLip2Backbone,
    SigLip2Grid37Backbone,
)


def test_grid37_control_is_pinned_to_37_patches():
    assert SigLip2Grid37Backbone.name == "siglip2_grid37"
    assert SigLip2Grid37Backbone.forced_patch_grid == 37


def test_position_grid_is_interpolated_to_the_token_grid():
    source_grid, target_grid, width = 3, 5, 8
    position = torch.randn(1, source_grid * source_grid, width)
    holder = SimpleNamespace(visual=SimpleNamespace(trunk=SimpleNamespace(
        num_prefix_tokens=0, pos_embed=position)))
    tokens = torch.zeros(2, target_grid * target_grid, width)

    result = SigLip2Backbone._interpolate_position_embedding(holder, tokens)

    assert result.shape == tokens.shape
    assert torch.isfinite(result).all()
    # The same interpolated positional grid is broadcast across the batch.
    assert torch.equal(result[0], result[1])
