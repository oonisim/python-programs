"""Tests that detect common Transformer training pitfalls.

These tests are intentionally strict and will fail if regressions reintroduce
well-known training gotchas (padding masks, BOS/EOS handling, eval/train mode).
They are based on widely documented pitfalls in Transformer training practices.
"""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from model.constant import LABEL_IGNORE_VALUE  # noqa: E402
from model.lm import LanguageModel  # noqa: E402
from model.model import Transformer  # noqa: E402
from training.loader_translation import translation_collate_fn  # noqa: E402


def test_transformer_generate_restores_training_mode():
    """Generate should not permanently switch the model into eval mode.

    If generate() flips to eval mode without restoring the previous state, any
    subsequent training step will silently run with dropout disabled.
    """
    model = Transformer()
    model.train()
    assert model.training is True

    x = torch.zeros(1, 3, dtype=torch.long)
    _ = model.generate(x=x, start_token=1, end_token=2, max_length=4)

    # The model should still be in training mode after generation.
    assert model.training is True


def test_transformer_evaluate_restores_training_mode():
    """Evaluate should not permanently switch the model into eval mode."""
    model = Transformer()
    model.train()
    assert model.training is True

    x = torch.zeros(1, 3, dtype=torch.long)
    y = torch.zeros(1, 3, dtype=torch.long)
    _ = model.evaluate(x=x, y=y)

    # The model should still be in training mode after evaluation.
    assert model.training is True


def test_language_model_generate_restores_training_mode():
    """LM generate should not permanently switch the model into eval mode."""
    model = LanguageModel(
        vocab_size=32,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        max_seq_len=8,
    )
    model.train()
    assert model.training is True

    prompt = torch.zeros(3, dtype=torch.long)
    _ = model.generate(prompt=prompt, max_length=6, top_k=5)

    # The model should still be in training mode after generation.
    assert model.training is True


def test_translation_padding_does_not_mask_real_eos_when_pad_equals_eos():
    """EOS tokens should not be masked out when pad_token_id == eos_token_id.

    Many GPT-style tokenizers reuse EOS as PAD. Masking by equality would
    incorrectly drop true EOS positions from loss computation.
    """
    pad_id = 0
    eos_id = 0  # Simulate pad == eos

    batch = [
        {"source_ids": torch.tensor([1, 2, 3]), "target_ids": torch.tensor([4, eos_id])},
        {"source_ids": torch.tensor([5, 6]), "target_ids": torch.tensor([7, eos_id])},
    ]

    out = translation_collate_fn(
        batch=batch,
        source_pad_id=pad_id,
        target_pad_id=pad_id,
    )
    target_ids = out["target_ids"]

    # The final EOS token should remain EOS, not be replaced by LABEL_IGNORE_VALUE.
    assert (target_ids[:, -1] == eos_id).all()
    assert (target_ids[:, -1] != LABEL_IGNORE_VALUE).all()


def test_translation_source_pad_mask_does_not_mask_real_eos_when_pad_equals_eos():
    """Source padding mask should only hide artificial padding.

    When pad_token_id == eos_token_id, the only reliable way to identify padding
    is by sequence length (positions beyond the original length). This test
    asserts that real tokens (including EOS) are not masked, and only the
    padded positions are masked.
    """
    pad_id = 0
    eos_id = 0  # Simulate pad == eos

    # Two source sequences of different lengths:
    # - seq0 length 3, last token is real EOS
    # - seq1 length 2, last token is real EOS, and padding is added to length 3
    batch = [
        {"source_ids": torch.tensor([1, 2, eos_id]), "target_ids": torch.tensor([3, 4])},
        {"source_ids": torch.tensor([5, eos_id]), "target_ids": torch.tensor([6, 7])},
    ]

    out = translation_collate_fn(
        batch=batch,
        source_pad_id=pad_id,
        target_pad_id=pad_id,
    )
    source_pad_mask = out["source_pad_mask"]

    # Expected mask by length (not by token value):
    # seq0 length 3 -> no padding
    # seq1 length 2 -> last position is padding
    expected_mask = torch.tensor(
        [[False, False, False],
         [False, False, True]],
        dtype=torch.bool,
    )

    assert torch.equal(source_pad_mask, expected_mask)
