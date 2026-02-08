"""Tests for decoder.py."""

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from transformer.decoder import DecodeLayer, Decoder  # noqa: E402


def test_decoder_output_shape_with_memory():
    """Decoder should return (B, T, D) when memory is provided."""
    # Input: target indices y and encoder memory with matching D.
    batch = 2
    time_steps = 4
    d_model = 8
    decoder = Decoder(
        vocabulary_size=32,
        num_layers=1,
        num_heads=2,
        d_model=d_model,
        max_time_steps=8,
        d_ff=16,
    )
    y = torch.zeros(batch, time_steps, dtype=torch.long)
    memory = torch.zeros(batch, time_steps, d_model)

    # Expected: decoder output shape (B, T, D) with cross-attention.
    out = decoder(y=y, memory=memory)
    assert out.shape == (batch, time_steps, d_model)


def test_decoder_output_shape_without_memory():
    """Decoder should return (B, T, D) without memory (decoder-only mode)."""
    # Input: target indices y with no memory for decoder-only mode.
    batch = 1
    time_steps = 3
    d_model = 8
    decoder = Decoder(
        vocabulary_size=16,
        num_layers=1,
        num_heads=2,
        d_model=d_model,
        max_time_steps=8,
        d_ff=16,
    )
    y = torch.zeros(batch, time_steps, dtype=torch.long)

    # Expected: output shape (B, T, D) without cross-attention.
    out = decoder(y=y, memory=None)
    assert out.shape == (batch, time_steps, d_model)


def test_decode_layer_invalid_memory_shape():
    """DecodeLayer should assert when memory has wrong feature dimension."""
    # Input: memory has D=7 but layer expects D=8.
    layer = DecodeLayer(i_layer=0, num_heads=2, d_model=8, max_time_steps=8)
    x = torch.zeros(1, 2, 8)
    memory = torch.zeros(1, 2, 7)

    # Expected: runtime error from LayerNorm due to feature mismatch.
    with pytest.raises(RuntimeError):
        layer(x=x, memory=memory)


def test_decoder_invalid_input_shape():
    """Decoder should assert when target indices are not 2D."""
    # Input: 3D target indices are invalid for decoder input.
    decoder = Decoder(vocabulary_size=16, num_layers=1, num_heads=2, d_model=8)
    y = torch.zeros(1, 2, 3)

    # Expected: assertion error because y must be 2D (B, T).
    with pytest.raises(AssertionError):
        decoder(y=y, memory=None)


def test_decode_layer_norm_variance_is_one():
    """DecodeLayer LayerNorm should yield unit variance per token."""
    # Input: random tensor to avoid zero variance edge case.
    torch.manual_seed(0)
    layer = DecodeLayer(i_layer=0, num_heads=2, d_model=8, max_time_steps=8)
    x = torch.randn(2, 3, 8)

    # Expected: per-token mean ~0 and variance ~1 on last dimension.
    y = layer.layer_norm_input(x)
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, correction=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)
