"""Tests for model.py Transformer."""

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from model.constant import NUM_CLASSES  # noqa: E402
from model.model import Transformer  # noqa: E402


def test_transformer_forward_shape():
    """Transformer forward should return (B, T, V) log-probabilities."""
    # Input: encoder and decoder token indices with same (B, T).
    model = Transformer()
    batch = 2
    time_steps = 3
    x = torch.zeros(batch, time_steps, dtype=torch.long)
    y = torch.zeros(batch, time_steps, dtype=torch.long)

    # Expected: log-probabilities shape (B, T, V) with V=NUM_CLASSES.
    out = model.forward(x=x, y=y)
    assert out.shape == (batch, time_steps, NUM_CLASSES)


def test_transformer_forward_invalid_x_shape():
    """Transformer forward should raise ValueError for invalid x shape."""
    # Input: x is 3D, which is invalid for encoder indices.
    model = Transformer()
    x = torch.zeros(1, 2, 3)
    y = torch.zeros(1, 2, dtype=torch.long)

    # Expected: ValueError because x must be 2D (B, T).
    with pytest.raises(ValueError):
        model.forward(x=x, y=y)


def test_transformer_forward_invalid_y_shape():
    """Transformer forward should raise ValueError for invalid y shape."""
    # Input: y is 3D, which is invalid for decoder indices.
    model = Transformer()
    x = torch.zeros(1, 2, dtype=torch.long)
    y = torch.zeros(1, 2, 3)

    # Expected: ValueError because y must be 2D (B, T).
    with pytest.raises(ValueError):
        model.forward(x=x, y=y)


def test_transformer_generate_shape():
    """Generate should return (B, T) with T <= max_length."""
    # Input: encoder indices and explicit start/end tokens.
    model = Transformer()
    x = torch.zeros(1, 3, dtype=torch.long)

    # Expected: output batch size preserved and length <= max_length.
    out = model.generate(x=x, start_token=1, end_token=2, max_length=5)
    assert out.shape[0] == 1
    assert out.shape[1] <= 5
