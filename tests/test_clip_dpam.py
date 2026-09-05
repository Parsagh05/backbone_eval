"""Numerical parity for the functional AnomalyCLIP DPAM branch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bbeval.backbones.clip import _dpam_value_attention  # noqa: E402


def test_dpam_value_attention_matches_official_vv_formula():
    width, heads, length, batch = 16, 4, 7, 2
    block = torch.nn.Module()
    block.ln_1 = torch.nn.LayerNorm(width)
    block.attn = torch.nn.MultiheadAttention(width, heads, dropout=0.0)
    inputs = torch.randn(length, batch, width)

    actual = _dpam_value_attention(block, inputs)

    # Independent transcription of AnomalyCLIP_lib.Attention.forward: project
    # QKV, replace Q and K with V, softmax V-V attention, then output-project.
    normed = block.ln_1(inputs).permute(1, 0, 2)
    qkv = torch.nn.functional.linear(
        normed, block.attn.in_proj_weight, block.attn.in_proj_bias)
    value = qkv.reshape(batch, length, 3, heads, width // heads)
    value = value.permute(2, 0, 3, 1, 4)[2]
    weights = ((value @ value.transpose(-2, -1))
               * ((width // heads) ** -0.5)).softmax(dim=-1)
    expected = (weights @ value).transpose(1, 2).reshape(batch, length, width)
    expected = block.attn.out_proj(expected).permute(1, 0, 2)

    assert torch.allclose(actual, expected, atol=1e-6)
