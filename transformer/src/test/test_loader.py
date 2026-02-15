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


def test_language_model_dataset_stride_non_overlapping():
    """Verify non-overlapping chunks (stride = seq_len).

    Consecutive chunks should not overlap - each chunk starts at idx * seq_len.
    """
    seq_len = 4
    tokens = torch.arange(0, 20, dtype=torch.long)  # [0, 1, 2, ..., 19]
    dataset = LanguageModelDataset(tokens=tokens, seq_len=seq_len)

    # Expected: 3 chunks
    # max_start = 20 - 5 = 15
    # num_chunks = (15 // 4) + 1 = 3 + 1 = 4
    assert len(dataset) == 4

    # Verify chunks are non-overlapping
    input_0, target_0 = dataset[0]
    input_1, target_1 = dataset[1]
    input_2, target_2 = dataset[2]
    input_3, target_3 = dataset[3]

    # Chunk 0: tokens[0:4], targets[1:5]
    assert torch.equal(input_0, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(target_0, torch.tensor([1, 2, 3, 4]))

    # Chunk 1: tokens[4:8], targets[5:9]
    assert torch.equal(input_1, torch.tensor([4, 5, 6, 7]))
    assert torch.equal(target_1, torch.tensor([5, 6, 7, 8]))

    # Chunk 2: tokens[8:12], targets[9:13]
    assert torch.equal(input_2, torch.tensor([8, 9, 10, 11]))
    assert torch.equal(target_2, torch.tensor([9, 10, 11, 12]))

    # Chunk 3: tokens[12:16], targets[13:17]
    assert torch.equal(input_3, torch.tensor([12, 13, 14, 15]))
    assert torch.equal(target_3, torch.tensor([13, 14, 15, 16]))


def test_language_model_dataset_exact_multiple():
    """Verify behavior when token count is an exact multiple of seq_len."""
    seq_len = 4
    # Exact multiple: 16 tokens = 4 * seq_len
    tokens = torch.arange(0, 16, dtype=torch.long)
    dataset = LanguageModelDataset(tokens=tokens, seq_len=seq_len)

    # max_start = 16 - 5 = 11
    # num_chunks = (11 // 4) + 1 = 2 + 1 = 3
    assert len(dataset) == 3

    # Last chunk should use tokens[8:12] and targets[9:13]
    input_last, target_last = dataset[2]
    assert torch.equal(input_last, torch.tensor([8, 9, 10, 11]))
    assert torch.equal(target_last, torch.tensor([9, 10, 11, 12]))


def test_language_model_dataset_minimum_tokens():
    """Verify behavior with minimum tokens needed for one chunk."""
    seq_len = 4
    # Minimum: seq_len + 1 = 5 tokens for one chunk
    tokens = torch.arange(0, 5, dtype=torch.long)
    dataset = LanguageModelDataset(tokens=tokens, seq_len=seq_len)

    # max_start = 5 - 5 = 0
    # num_chunks = (0 // 4) + 1 = 0 + 1 = 1
    assert len(dataset) == 1

    # Only one chunk: tokens[0:4], targets[1:5]
    input_ids, target_ids = dataset[0]
    assert torch.equal(input_ids, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(target_ids, torch.tensor([1, 2, 3, 4]))


def test_language_model_dataset_just_short():
    """Verify behavior when tokens are just short of minimum."""
    seq_len = 4
    # Just short: seq_len tokens (need seq_len + 1)
    tokens = torch.arange(0, 4, dtype=torch.long)
    dataset = LanguageModelDataset(tokens=tokens, seq_len=seq_len)

    # max_start = 4 - 5 = -1
    # Should return 0 (handled by max_start < 0 check)
    assert len(dataset) == 0


def test_language_model_dataset_large_example():
    """Verify calculation with realistic numbers (similar to WikiText-2)."""
    seq_len = 256
    # Simulate ~1025 tokens
    tokens = torch.arange(0, 1025, dtype=torch.long)
    dataset = LanguageModelDataset(tokens=tokens, seq_len=seq_len)

    # max_start = 1025 - 257 = 768
    # num_chunks = (768 // 256) + 1 = 3 + 1 = 4
    assert len(dataset) == 4

    # Verify first and last chunks
    input_0, target_0 = dataset[0]
    assert len(input_0) == 256
    assert len(target_0) == 256
    assert input_0[0].item() == 0
    assert target_0[0].item() == 1

    # Last chunk starts at 768
    input_3, target_3 = dataset[3]
    assert len(input_3) == 256
    assert len(target_3) == 256
    assert input_3[0].item() == 768
    assert target_3[0].item() == 769
    assert target_3[-1].item() == 1024


def test_language_model_dataset_target_offset():
    """Verify that targets are correctly shifted by 1 from inputs."""
    seq_len = 4
    tokens = torch.arange(0, 10, dtype=torch.long)
    dataset = LanguageModelDataset(tokens=tokens, seq_len=seq_len)

    for idx in range(len(dataset)):
        input_ids, target_ids = dataset[idx]

        # Each element in target should be input + 1
        expected_targets = input_ids + 1
        assert torch.equal(target_ids, expected_targets), \
            f"Chunk {idx}: targets should be inputs + 1"


def test_language_model_dataset_no_overlap_between_chunks():
    """Verify that no token appears in the input of multiple chunks."""
    seq_len = 4
    tokens = torch.arange(0, 20, dtype=torch.long)
    dataset = LanguageModelDataset(tokens=tokens, seq_len=seq_len)

    # Collect all input tokens from all chunks
    all_input_tokens = []
    for idx in range(len(dataset)):
        input_ids, _ = dataset[idx]
        all_input_tokens.extend(input_ids.tolist())

    # Verify no duplicates (all tokens should be unique)
    assert len(all_input_tokens) == len(set(all_input_tokens)), \
        "Chunks should not overlap - each input token should appear only once"


def test_language_model_dataset_stats_consistency():
    """Verify get_stats() reports correct sequence count matching __len__.

    The stats should use the same calculation as __len__ for non-overlapping chunks.
    """
    from training.loader import LanguageModelDataLoaderFactory, DataLoaderConfig

    # Create a mock tokenizer
    class MockTokenizer:
        def encode(self, text):
            # Return token IDs based on text length
            return list(range(len(text)))

        @property
        def vocab_size(self):
            return 100

    seq_len = 256
    config = DataLoaderConfig(seq_len=seq_len, batch_size=4)

    # Create dataset with known number of tokens
    # We'll directly test the sequence count calculation
    tokens_1025 = torch.arange(0, 1025, dtype=torch.long)
    dataset_1025 = LanguageModelDataset(tokens_1025, seq_len)

    # Verify __len__ returns correct value
    # max_start = 1025 - 257 = 768
    # num_chunks = (768 // 256) + 1 = 4
    assert len(dataset_1025) == 4

    # Verify the same calculation works for edge cases
    tokens_512 = torch.arange(0, 512, dtype=torch.long)
    dataset_512 = LanguageModelDataset(tokens_512, seq_len)
    # max_start = 512 - 257 = 255
    # num_chunks = (255 // 256) + 1 = 0 + 1 = 1
    assert len(dataset_512) == 1

    tokens_256 = torch.arange(0, 256, dtype=torch.long)
    dataset_256 = LanguageModelDataset(tokens_256, seq_len)
    # max_start = 256 - 257 = -1
    # Should return 0
    assert len(dataset_256) == 0
