"""Data loading module for language model training.

This module provides data ingestion and feature engineering (tokenization)
for language modeling. The tokenizer is injected as a dependency for flexibility.

Architecture:
    Dataset (HuggingFace) -> Tokenization -> LanguageModelDataset -> DataLoader

Usage:
    from transformers import GPT2Tokenizer
    from scratch.loader import LanguageModelDataLoaderFactory

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    factory = LanguageModelDataLoaderFactory(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        tokenizer=tokenizer,
        seq_len=256,
        batch_size=32
    )

    train_loader = factory.get_train_loader()
    val_loader = factory.get_val_loader()
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from tokenization import Tokenizer


class LanguageModelDataset(Dataset):
    """Dataset for language modeling with non-overlapping block chunking.

    For language modeling, input and target are the same sequence shifted by 1:
    - Input:  [t0, t1, t2, ..., t_{n-1}]
    - Target: [t1, t2, t3, ..., t_n]

    The model learns to predict the next token given previous tokens.

    IMPORTANT - Block Chunking (stride = seq_len):
    -----------------------------------------------
    This implementation uses NON-OVERLAPPING chunks (stride = seq_len), which is
    standard for GPT-style LM training.

    Previous bug: Used stride=1 sliding window, creating ~N sequences (one per token).
    - With 118M tokens, seq_len=512: created 118M sequences → 3.6M batches/epoch
    - Result: ~32 days per epoch at 4,740 steps/hour

    Current fix: Block chunking with stride=seq_len creates ~N/seq_len sequences.
    - With 118M tokens, seq_len=512: creates 230K sequences → 7.2K batches/epoch
    - Result: ~1.5 hours per epoch (512x faster)

    Why block chunking?
    - Efficient: Processes each token exactly once per epoch
    - Standard practice: Used by GPT-2, GPT-3, and all modern LM training
    - No data correlation: Adjacent batches come from different text regions
    - Sliding window (stride=1) is only used for evaluation, not training
    """

    def __init__(self, tokens: Tensor, seq_len: int):
        """Initialize dataset.

        Args:
            tokens: All tokens as a single 1D tensor.
            seq_len: Sequence length for each sample.
        """
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        """Number of complete sequences available (non-overlapping chunks).

        Uses block chunking with stride = seq_len (no overlap).
        Each sample needs seq_len + 1 tokens total:
          - input_ids:  tokens[start : start + seq_len]         (seq_len tokens)
          - target_ids: tokens[start + 1 : start + seq_len + 1] (seq_len tokens)

        Valid start positions must satisfy: start + seq_len + 1 <= len(tokens)
        Max valid start: max_start = len(tokens) - (seq_len + 1)
        Number of non-overlapping chunks: (max_start // seq_len) + 1

        Example with N=1025 tokens, seq_len=256:
            max_start = 1025 - 257 = 768
            chunks = (768 // 256) + 1 = 4
            - Chunk 0: start=0,   tokens[0:256],   targets[1:257]     ✓
            - Chunk 1: start=256, tokens[256:512], targets[257:513]   ✓
            - Chunk 2: start=512, tokens[512:768], targets[513:769]   ✓
            - Chunk 3: start=768, tokens[768:1024], targets[769:1025] ✓
        """
        n = len(self.tokens)
        max_start = n - (self.seq_len + 1)
        if max_start < 0:
            return 0
        return (max_start // self.seq_len) + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get input-target pair from non-overlapping block chunks.

        Args:
            idx: Chunk index (not token position). Valid range: [0, __len__())

        Returns:
            Tuple of (input_ids, target_ids) where target is shifted by 1.
            Each chunk starts at position (idx * seq_len).
        """
        start = idx * self.seq_len
        input_ids = self.tokens[start : start + self.seq_len]
        target_ids = self.tokens[start + 1 : start + self.seq_len + 1]
        return input_ids, target_ids


@dataclass
class DataLoaderConfig:
    """Configuration for data loading."""
    seq_len: int = 256
    batch_size: int = 32
    num_workers: int = 0
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None


class LanguageModelDataLoaderFactory:
    """Factory for creating language model DataLoaders.

    Handles the complete pipeline:
    1. Load raw dataset from HuggingFace
    2. Tokenize text using injected tokenizer
    3. Create LanguageModelDataset
    4. Return DataLoader

    The tokenizer is injected as a dependency, allowing flexibility in
    choosing different tokenization strategies (GPT-2 BPE, BERT WordPiece, etc.).
    """

    def __init__(
            self,
            dataset_name: str,
            tokenizer: Tokenizer,
            config: Optional[DataLoaderConfig] = None,
            dataset_config: Optional[str] = None,
    ):
        """Initialize the factory.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "wikitext").
            tokenizer: Tokenizer instance with encode() method.
            config: Data loading configuration.
            dataset_config: Dataset configuration (e.g., "wikitext-2-raw-v1").
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.config = config or DataLoaderConfig()

        self._raw_dataset = None
        self._train_tokens = None
        self._val_tokens = None

    def _load_raw_dataset(self) -> None:
        """Load raw dataset from HuggingFace."""
        if self._raw_dataset is None:
            self._raw_dataset = load_dataset(
                self.dataset_name,
                self.dataset_config
            )

    def _tokenize_split(
            self,
            split: str,
            max_samples: Optional[int] = None
    ) -> Tensor:
        """Tokenize a dataset split into a single token sequence.

        Args:
            split: Dataset split name ("train", "validation", "test").
            max_samples: Maximum number of samples to use (None for all).

        Returns:
            1D tensor of all token IDs concatenated.
        """
        self._load_raw_dataset()
        data = self._raw_dataset[split]

        if max_samples is not None:
            data = data.select(range(min(max_samples, len(data))))

        all_tokens = []
        for item in data:
            text = item["text"]
            if text.strip():  # Skip empty lines
                tokens = self.tokenizer.encode(text)
                # Chunk long token sequences to avoid warnings/issues
                # when individual articles produce very long sequences
                # (e.g., 1063 tokens from a long WikiText article)
                if len(tokens) > self.config.seq_len:
                    # Split into chunks of seq_len to respect model limits
                    for i in range(0, len(tokens), self.config.seq_len):
                        chunk = tokens[i:i + self.config.seq_len]
                        all_tokens.extend(chunk)
                else:
                    all_tokens.extend(tokens)

        return torch.tensor(all_tokens, dtype=torch.long)

    def get_train_loader(self) -> DataLoader:
        """Create training DataLoader.

        Returns:
            DataLoader for training data with shuffling enabled.
        """
        if self._train_tokens is None:
            self._train_tokens = self._tokenize_split(
                "train",
                self.config.max_train_samples
            )

        dataset = LanguageModelDataset(
            self._train_tokens,
            self.config.seq_len
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True  # Drop final incomplete batch to keep batch sizes consistent
        )

    def get_val_loader(self) -> DataLoader:
        """Create validation DataLoader.

        Returns:
            DataLoader for validation data without shuffling.
        """
        if self._val_tokens is None:
            self._val_tokens = self._tokenize_split(
                "validation",
                self.config.max_val_samples
            )

        dataset = LanguageModelDataset(
            self._val_tokens,
            self.config.seq_len
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

    def get_test_loader(self) -> DataLoader:
        """Create test DataLoader.

        Returns:
            DataLoader for test data without shuffling.
        """
        test_tokens = self._tokenize_split("test")

        dataset = LanguageModelDataset(
            test_tokens,
            self.config.seq_len
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

    def get_stats(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dictionary with token counts and sequence counts.
        """
        self._load_raw_dataset()

        stats = {
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "raw_train_samples": len(self._raw_dataset["train"]),
            "raw_val_samples": len(self._raw_dataset["validation"]),
            "raw_test_samples": len(self._raw_dataset["test"]),
        }

        if self._train_tokens is not None:
            stats["train_tokens"] = len(self._train_tokens)
            # Calculate number of non-overlapping chunks using same formula as __len__
            max_start = len(self._train_tokens) - (self.config.seq_len + 1)
            stats["train_sequences"] = 0 if max_start < 0 else (max_start // self.config.seq_len) + 1

        if self._val_tokens is not None:
            stats["val_tokens"] = len(self._val_tokens)
            # Calculate number of non-overlapping chunks using same formula as __len__
            max_start = len(self._val_tokens) - (self.config.seq_len + 1)
            stats["val_sequences"] = 0 if max_start < 0 else (max_start // self.config.seq_len) + 1

        return stats


if __name__ == "__main__":
    print(__doc__)
