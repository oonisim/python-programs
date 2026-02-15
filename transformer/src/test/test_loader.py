"""Tests for loader.py datasets."""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from training.loader import LanguageModelDataset  # noqa: E402


def test_language_model_dataset_shapes():
    """LanguageModelDataset should return input/target sequences of seq_len."""
    # Input: tokens length = seq_len + 2 to allow one sample.
    seq_len = 4
    tokens = torch.arange(0, seq_len + 2, dtype=torch.long)
    dataset = LanguageModelDataset(tokens=tokens, seq_len=seq_len)

    # Expected: input and target are both length seq_len.
    input_ids, target_ids = dataset[0]
    assert input_ids.shape[0] == seq_len
    assert target_ids.shape[0] == seq_len


def test_language_model_dataset_len_non_negative():
    """Dataset length should not be negative when tokens are short."""
    # Input: fewer tokens than seq_len should yield zero samples.
    seq_len = 8
    tokens = torch.arange(0, 4, dtype=torch.long)
    dataset = LanguageModelDataset(tokens=tokens, seq_len=seq_len)

    # Expected: no samples because tokens are insufficient.
    assert len(dataset) == 0
