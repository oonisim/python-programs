"""Tests for encoder.py."""

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from model.encoder import Encoder, EncodeLayer  # noqa: E402

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
    """Encoder should return (B, T, D) for valid embedded input."""
    # Input: embedded vectors with batch=2, time=4, model_dim=8.
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
    x = torch.randn(batch, time_steps, d_model)

    # Expected: output shape (B, T, D) from encoder stack.
    out = encoder(x)
    assert out.shape == (batch, time_steps, d_model)


def test_encoder_invalid_input_shape():
    """Encoder should assert when input tensor is not 3D."""
    # Input: 2D tensor is invalid for encoder (expects embedded (B, T, D)).
    encoder = Encoder(vocabulary_size=16, num_layers=1, num_heads=2, d_model=8)
    x = torch.zeros(1, 2)

    # Expected: assertion error because input must be 3D (B, T, D).
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
    print(f"BERT tokens for 'hello world': {tokens}")

    d_model = 8
    # Expected: encoder output shape (B, T, D) with variance ~1 at LayerNorm.
    encoder = Encoder(
        vocabulary_size=max(tokens) + 1,
        num_layers=1,
        num_heads=2,
        d_model=d_model,
        max_time_steps=16,
        d_ff=16,
    )
    # Encoder now expects embedded (B, T, D) tensors
    x = torch.randn(1, len(tokens), d_model)
    out = encoder(x)
    assert out.shape == (1, len(tokens), d_model)
    normed = encoder.layers[0].layer_norm_input(out)
    var = normed.var(dim=-1, correction=0)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


def test_encoder_output_shape_with_xlm_roberta_tokens():
    """Encoder output shape matches (B, T, D) with XLM-RoBERTa SentencePiece tokens."""
    # Input: XLM-RoBERTa (SentencePiece BPE) tokenizer used in multilingual
    # Sentence Transformers (e.g., BGE-M3, multilingual-e5).
    tokenizer = _load_tokenizer("xlm-roberta-base")
    tokens = tokenizer.encode("hello world", add_special_tokens=True)
    print(f"XLM-RoBERTa SentencePiece tokens for 'hello world': {tokens}")

    d_model = 8
    # Expected: encoder output shape (B, T, D) with variance ~1 at LayerNorm.
    encoder = Encoder(
        vocabulary_size=max(tokens) + 1,
        num_layers=1,
        num_heads=2,
        d_model=d_model,
        max_time_steps=16,
        d_ff=16,
    )
    # Encoder now expects embedded (B, T, D) tensors
    x = torch.randn(1, len(tokens), d_model)
    out = encoder(x)
    assert out.shape == (1, len(tokens), d_model)
    normed = encoder.layers[0].layer_norm_input(out)
    var = normed.var(dim=-1, correction=0)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-4)
