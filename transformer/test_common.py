"""Tests for common.py utilities and layers."""

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from transformer.common import (  # noqa: E402
    LayerNormalization,
    PositionalEncoding,
    ScaledDotProductAttention,
    mask,
    split,
)


def test_split_shape():
    """Split should reshape to (B, H, T, d_k) with preserved batch and time."""
    # Input: batch=2, time=3, model_dim=8 with 2 heads.
    batch = 2
    time_steps = 3
    dim = 8
    heads = 2
    x = torch.zeros(batch, time_steps, dim)

    # Expected: reshape into (B, H, T, d_k) where d_k = D / H = 4.
    out = split(x, heads)
    assert out.shape == (batch, heads, time_steps, dim // heads)


def test_mask_truncates_and_masks_future():
    """Mask should truncate larger masks and set future positions to -inf."""
    # Input: similarities for T=2, but mask is T=4 upper-triangular.
    similarities = torch.zeros(1, 1, 2, 2)
    big_mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()

    # Expected: mask is truncated to 2x2 and future position becomes -inf.
    masked = mask(similarities, big_mask)
    assert masked.shape == (1, 1, 2, 2)
    assert torch.isinf(masked[0, 0, 0, 1])


def test_mask_future_keys_zero_after_softmax():
    """Mask should zero out future key probabilities after softmax."""
    # Input: q->k similarities with higher value for future key at [0, 1].
    similarities = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    future_mask = torch.tensor([[False, True], [False, False]])

    # Expected: masked similarity becomes -inf and softmax probability is 0.
    masked = mask(similarities, future_mask)
    probabilities = torch.softmax(masked, dim=-1)
    assert masked[0, 0, 0, 1] == float("-inf")
    assert probabilities[0, 0, 0, 1] == 0.0


def test_scaled_dot_product_attention_invalid_shape():
    """ScaledDotProductAttention should reject non-4D query tensors."""
    # Input: invalid q/k/v without head dimension.
    attention = ScaledDotProductAttention(do_mask=False, max_time_steps=4)
    q = torch.zeros(1, 2, 4)  # Invalid: missing head dimension.
    k = torch.zeros(1, 2, 4)
    v = torch.zeros(1, 2, 4)

    # Expected: runtime error due to incorrect dimensionality.
    with pytest.raises(RuntimeError):
        attention(q=q, k=k, v=v)


def test_scaled_dot_product_attention_applies_mask():
    """ScaledDotProductAttention should zero out future keys when masked."""
    # Input: 5-token sequence with do_mask=True, values are all ones.
    attention = ScaledDotProductAttention(do_mask=True, max_time_steps=5)
    q = torch.ones(1, 1, 5, 1)
    k = torch.ones(1, 1, 5, 1)
    v = torch.ones(1, 1, 5, 1)

    # Output: probabilities has shape (B, H, Tq, Tk). Index [0,0,0,4] means:
    # batch=0, head=0, query_pos=0 attending to key_pos=4 (a future key).
    # Expected: for each query position t, only keys [0..t] are allowed.
    # Because q/k are all ones, unmasked similarities are equal, so softmax
    # yields a uniform distribution over allowed keys: 1 / (t + 1).
    _, probabilities = attention(q=q, k=k, v=v, return_similarities=True)
    print(f"ScaledDotProductAttention probabilities: {probabilities.tolist()}")
    for t in range(5):
        expected = torch.full((t + 1,), 1.0 / (t + 1))
        actual_allowed = probabilities[0, 0, t, : t + 1]
        actual_future = probabilities[0, 0, t, t + 1 :]
        assert torch.allclose(actual_allowed, expected, atol=1e-6)
        assert torch.allclose(actual_future, torch.zeros(5 - t - 1), atol=0.0)


def test_positional_encoding_shape():
    """PositionalEncoding should return (1, T, D) matching input time and dim."""
    # Input: token embeddings with T=5 and D=6.
    d_model = 6
    max_time = 8
    x = torch.zeros(2, 5, d_model)

    # Expected: (1, T, D) so it can broadcast over batch.
    encoding = PositionalEncoding(max_time_steps=max_time, d_model=d_model)
    out = encoding(x)
    assert out.shape == (1, 5, d_model)


def test_positional_encoding_values():
    """PositionalEncoding should match the sin/cos formula for small cases."""
    # Input: small d_model to validate exact sin/cos values at positions 0 and 1.
    d_model = 4
    max_time = 3
    x = torch.zeros(1, 2, d_model)
    encoding = PositionalEncoding(max_time_steps=max_time, d_model=d_model)

    # Expected: at position 0, sin(0)=0 and cos(0)=1 for all frequencies.
    out = encoding(x)
    assert torch.allclose(out[0, 0, 0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(out[0, 0, 1], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(out[0, 0, 2], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(out[0, 0, 3], torch.tensor(1.0), atol=1e-6)

    # Expected: at position 1, value equals sin(1 * div_term) and cos(1 * div_term).
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) *
        (-1 * (torch.log(torch.tensor(10000.0)) / d_model))
    )
    expected_sin = torch.sin(torch.tensor(1.0) * div_term)
    expected_cos = torch.cos(torch.tensor(1.0) * div_term)
    assert torch.allclose(out[0, 1, 0], expected_sin[0], atol=1e-6)
    assert torch.allclose(out[0, 1, 1], expected_cos[0], atol=1e-6)
    assert torch.allclose(out[0, 1, 2], expected_sin[1], atol=1e-6)
    assert torch.allclose(out[0, 1, 3], expected_cos[1], atol=1e-6)


def test_layer_normalization_variance_is_one():
    """LayerNormalization should produce unit variance per token."""
    # Input: random tensor to avoid zero variance edge case.
    torch.manual_seed(0)
    x = torch.randn(4, 3, 6)
    layer = LayerNormalization(features=6)

    # Expected: per-token mean ~0 and variance ~1 on last dimension.
    y = layer(x)
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, correction=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)
