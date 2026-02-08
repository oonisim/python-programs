"""Tests for encoder.py."""

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from transformer.encoder import Encoder, EncodeLayer  # noqa: E402

transformers = pytest.importorskip("transformers")


def _load_tokenizer(name: str):
    """Load tokenizer from local cache or skip if unavailable."""
    try:
        return transformers.AutoTokenizer.from_pretrained(
            name,
            local_files_only=True,
        )
    except OSError as exc:
        pytest.skip(f"Tokenizer not cached locally: {name}. {exc}")


def test_encoder_output_shape():
    """Encoder should return (B, T, D) for valid input indices."""
    # Input: token indices with batch=2, time=4, model_dim=8.
    batch = 2
    time_steps = 4
    d_model = 8
    encoder = Encoder(
        vocabulary_size=32,
        num_layers=1,
        num_heads=2,
        d_model=d_model,
        max_time_steps=8,
        d_ff=16,
    )
    x = torch.zeros(batch, time_steps, dtype=torch.long)

    # Expected: embedding output shape (B, T, D) from encoder stack.
    out = encoder(x)
    assert out.shape == (batch, time_steps, d_model)


def test_encoder_invalid_input_shape():
    """Encoder should assert when input tensor is not 2D."""
    # Input: 3D tensor is invalid for encoder indices.
    encoder = Encoder(vocabulary_size=16, num_layers=1, num_heads=2, d_model=8)
    x = torch.zeros(1, 2, 3)

    # Expected: assertion error because input must be 2D (B, T).
    with pytest.raises(AssertionError):
        encoder(x)


def test_encode_layer_norm_variance_is_one():
    """EncodeLayer LayerNorm should yield unit variance per token."""
    # Input: random tensor to avoid zero variance edge case.
    torch.manual_seed(0)
    layer = EncodeLayer(i_layer=0, num_heads=2, d_model=8, max_time_steps=8)
    x = torch.randn(2, 3, 8)

    # Expected: per-token mean ~0 and variance ~1 on last dimension.
    y = layer.layer_norm_input(x)
    mean = y.mean(dim=-1)
    var = y.var(dim=-1, correction=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_encoder_output_shape_with_bert_tokens():
    """Encoder output shape matches (B, T, D) with BERT token inputs."""
    # Input: BERT tokenizer with text "hello world" to generate token IDs.
    tokenizer = _load_tokenizer("bert-base-uncased")
    tokens = tokenizer.encode("hello world", add_special_tokens=True)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    print(f"BERT tokens for 'hello world': {tokens}")

    # Expected: encoder output shape (B, T, D) with variance ~1 at LayerNorm.
    encoder = Encoder(
        vocabulary_size=max(tokens) + 1,
        num_layers=1,
        num_heads=2,
        d_model=8,
        max_time_steps=16,
        d_ff=16,
    )
    out = encoder(input_ids)
    assert out.shape == (1, input_ids.shape[1], 8)
    normed = encoder.layers[0].layer_norm_input(out)
    var = normed.var(dim=-1, correction=0)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_encoder_output_shape_with_gpt_tokens():
    """Encoder output shape matches (B, T, D) with GPT token inputs."""
    # Input: GPT-2 BPE tokenizer with text "hello world!" to generate token IDs.
    tokenizer = _load_tokenizer("gpt2")
    tokens = tokenizer.encode("hello world!", add_special_tokens=True)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    print(f"GPT-2 BPE tokens for 'hello world!': {tokens}")

    # Expected: encoder output shape (B, T, D) with variance ~1 at LayerNorm.
    encoder = Encoder(
        vocabulary_size=max(tokens) + 1,
        num_layers=1,
        num_heads=2,
        d_model=8,
        max_time_steps=16,
        d_ff=16,
    )
    out = encoder(input_ids)
    assert out.shape == (1, input_ids.shape[1], 8)
    normed = encoder.layers[0].layer_norm_input(out)
    var = normed.var(dim=-1, correction=0)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)
