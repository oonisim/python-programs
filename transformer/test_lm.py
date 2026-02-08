"""Tests for lm.py LanguageModel."""

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from transformer.lm import LanguageModel  # noqa: E402


def test_language_model_forward_shape():
    """LanguageModel forward should return (B, T, V) log-probabilities."""
    # Input: token indices with batch=2, time=5, vocab=50.
    model = LanguageModel(
        vocab_size=50,
        d_model=16,
        num_heads=4,
        num_layers=1,
        d_ff=32,
        max_seq_len=8,
    )
    x = torch.zeros(2, 5, dtype=torch.long)

    # Expected: log-probabilities shape (B, T, V) with V=vocab_size.
    out = model.forward(x)
    assert out.shape == (2, 5, 50)


def test_language_model_generate_shape():
    """Generate should return (B, T) with T == max_length when no EOS set."""
    # Input: prompt length=3, no end_token so loop runs full length.
    model = LanguageModel(
        vocab_size=20,
        d_model=8,
        num_heads=2,
        num_layers=1,
        d_ff=16,
        max_seq_len=8,
    )
    prompt = torch.zeros(3, dtype=torch.long)

    # Expected: output length equals max_length for single batch.
    out = model.generate(prompt=prompt, max_length=6, top_k=5)
    assert out.shape == (1, 6)


def test_language_model_generate_invalid_prompt():
    """Generate should assert when prompt has invalid dimensions."""
    # Input: 3D prompt is invalid; generate will call forward and assert.
    model = LanguageModel(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        num_layers=1,
        d_ff=16,
        max_seq_len=8,
    )
    prompt = torch.zeros(1, 2, 3)

    # Expected: assertion error because forward requires 2D input.
    with pytest.raises(AssertionError):
        model.generate(prompt=prompt, max_length=5)
