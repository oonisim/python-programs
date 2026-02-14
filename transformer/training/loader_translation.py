"""Data loading module for translation model training.

This module provides data ingestion and feature engineering (tokenization)
for sequence-to-sequence translation. Source and target tokenizers are
injected as dependencies for flexibility.

Architecture:
    Dataset (HuggingFace) -> Tokenization -> TranslationDataset -> DataLoader

Usage:
    from transformers import GPT2Tokenizer
    from loader_translation import TranslationDataLoaderFactory, DataLoaderConfig

    src_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tgt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    factory = TranslationDataLoaderFactory(
        dataset_name="opus_books",
        dataset_config="en-es",
        source_tokenizer=src_tokenizer,
        target_tokenizer=tgt_tokenizer,
        source_lang="en",
        target_lang="es",
        config=DataLoaderConfig(max_seq_len=128, batch_size=32)
    )

    train_loader = factory.get_train_loader()
    val_loader = factory.get_val_loader()
"""
from dataclasses import dataclass
from typing import Optional, Protocol

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


class Tokenizer(Protocol):
    """Protocol for tokenizer interface (duck typing)."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    @property
    def pad_token_id(self) -> Optional[int]:
        """Padding token ID."""
        ...

    @property
    def eos_token_id(self) -> Optional[int]:
        """End of sequence token ID."""
        ...

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        ...


@dataclass
class DataLoaderConfig:
    """Configuration for translation data loading."""
    max_seq_len: int = 128
    batch_size: int = 32
    num_workers: int = 0
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    val_split_ratio: float = 0.1


class TranslationDataset(Dataset):
    """Dataset for translation (sequence-to-sequence).

    Each item is a pair of (source_ids, target_ids) tensors.
    Returns dicts with "source_ids" and "target_ids" keys to match
    the Trainer's expected batch format.
    """

    def __init__(self, pairs: list[tuple[Tensor, Tensor]]):
        """Initialize dataset.

        Args:
            pairs: List of (source_ids, target_ids) tensor pairs.
        """
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        source_ids, target_ids = self.pairs[idx]
        return {"source_ids": source_ids, "target_ids": target_ids}


def translation_collate_fn(
        batch: list[dict[str, Tensor]],
        source_pad_id: int,
        target_pad_id: int
) -> dict[str, Tensor]:
    """Collate function that pads source and target to max length in batch.

    Args:
        batch: List of dicts with "source_ids" and "target_ids".
        source_pad_id: Padding token ID for source sequences.
        target_pad_id: Padding token ID for target sequences.

    Returns:
        Dict with padded "source_ids" (B, Ts) and "target_ids" (B, Tt).
    """
    source_seqs = [item["source_ids"] for item in batch]
    target_seqs = [item["target_ids"] for item in batch]

    source_padded = pad_sequence(
        source_seqs, batch_first=True, padding_value=source_pad_id
    )
    target_padded = pad_sequence(
        target_seqs, batch_first=True, padding_value=target_pad_id
    )

    return {"source_ids": source_padded, "target_ids": target_padded}


class TranslationDataLoaderFactory:
    """Factory for creating translation DataLoaders.

    Handles the complete pipeline:
    1. Load raw dataset from HuggingFace
    2. Tokenize source/target with respective tokenizers
    3. Truncate and append EOS
    4. Create TranslationDataset
    5. Return DataLoader with padding collation

    Source and target tokenizers are injected as dependencies.
    """

    def __init__(
            self,
            dataset_name: str,
            dataset_config: str,
            source_tokenizer: Tokenizer,
            target_tokenizer: Tokenizer,
            source_lang: str,
            target_lang: str,
            config: Optional[DataLoaderConfig] = None,
    ):
        """Initialize the factory.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "opus_books").
            dataset_config: Dataset config (e.g., "en-es").
            source_tokenizer: Tokenizer for source language.
            target_tokenizer: Tokenizer for target language.
            source_lang: Source language code (e.g., "en").
            target_lang: Target language code (e.g., "es").
            config: Data loading configuration.
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.config = config or DataLoaderConfig()

        self._raw_dataset = None
        self._train_pairs: Optional[list[tuple[Tensor, Tensor]]] = None
        self._val_pairs: Optional[list[tuple[Tensor, Tensor]]] = None

    def _load_raw_dataset(self) -> None:
        """Load raw dataset from HuggingFace and split into train/val."""
        if self._raw_dataset is not None:
            return

        dataset = load_dataset(self.dataset_name, self.dataset_config)

        # opus_books only has "train" split; split into train/val
        if "validation" in dataset:
            self._raw_dataset = dataset
        else:
            split = dataset["train"].train_test_split(
                test_size=self.config.val_split_ratio, seed=42
            )
            self._raw_dataset = {
                "train": split["train"],
                "validation": split["test"],
            }

    def _tokenize_pairs(
            self,
            split: str,
            max_samples: Optional[int] = None
    ) -> list[tuple[Tensor, Tensor]]:
        """Tokenize paired sequences from a dataset split.

        Args:
            split: Dataset split name ("train" or "validation").
            max_samples: Maximum number of pairs to use.

        Returns:
            List of (source_ids, target_ids) tensor pairs.
        """
        self._load_raw_dataset()
        data = self._raw_dataset[split]

        if max_samples is not None:
            data = data.select(range(min(max_samples, len(data))))

        max_len = self.config.max_seq_len
        src_eos = self.source_tokenizer.eos_token_id
        tgt_eos = self.target_tokenizer.eos_token_id

        pairs = []
        for item in data:
            translation = item["translation"]
            src_text = translation[self.source_lang]
            tgt_text = translation[self.target_lang]

            if not src_text.strip() or not tgt_text.strip():
                continue

            src_ids = self.source_tokenizer.encode(src_text)
            tgt_ids = self.target_tokenizer.encode(tgt_text)

            # Truncate to max_seq_len - 1 to leave room for EOS
            src_ids = src_ids[: max_len - 1] + [src_eos]
            tgt_ids = tgt_ids[: max_len - 1] + [tgt_eos]

            pairs.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

        return pairs

    def get_train_loader(self) -> DataLoader:
        """Create training DataLoader.

        Returns:
            DataLoader for training data with shuffling enabled.
        """
        if self._train_pairs is None:
            self._train_pairs = self._tokenize_pairs(
                "train", self.config.max_train_samples
            )

        dataset = TranslationDataset(self._train_pairs)
        src_pad = self.source_tokenizer.pad_token_id
        tgt_pad = self.target_tokenizer.pad_token_id

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=lambda batch: translation_collate_fn(
                batch, src_pad, tgt_pad
            ),
        )

    def get_val_loader(self) -> DataLoader:
        """Create validation DataLoader.

        Returns:
            DataLoader for validation data without shuffling.
        """
        if self._val_pairs is None:
            self._val_pairs = self._tokenize_pairs(
                "validation", self.config.max_val_samples
            )

        dataset = TranslationDataset(self._val_pairs)
        src_pad = self.source_tokenizer.pad_token_id
        tgt_pad = self.target_tokenizer.pad_token_id

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=lambda batch: translation_collate_fn(
                batch, src_pad, tgt_pad
            ),
        )

    def get_stats(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dictionary with pair counts and token statistics.
        """
        self._load_raw_dataset()

        stats = {
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "raw_train_pairs": len(self._raw_dataset["train"]),
            "raw_val_pairs": len(self._raw_dataset["validation"]),
        }

        if self._train_pairs is not None:
            stats["train_pairs"] = len(self._train_pairs)
            stats["train_source_tokens"] = sum(
                len(s) for s, _ in self._train_pairs
            )
            stats["train_target_tokens"] = sum(
                len(t) for _, t in self._train_pairs
            )

        if self._val_pairs is not None:
            stats["val_pairs"] = len(self._val_pairs)

        return stats
