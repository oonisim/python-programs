"""Tokenizer module for handling different tokenization strategies based on model type.
"""
from typing import (
    Optional,
    Protocol
)
import tiktoken


TOKENIZER_CONFIGS = {
    "gpt2": None,           # Uses HuggingFace GPT2Tokenizer
    "gpt4": "cl100k_base",  # tiktoken encoding for GPT-4
    "gpt4o": "o200k_base",  # tiktoken encoding for GPT-4o
}

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


# --------------------------------------------------------------------------------
# Tiktoken Adapter
# --------------------------------------------------------------------------------
class TiktokenAdapter(Tokenizer):
    """Adapter for tiktoken to match the Tokenizer protocol in loader.py."""

    def __init__(self, encoding_name: str):
        self._enc = tiktoken.get_encoding(encoding_name)
        self._eot_id = self._enc.eot_token

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eos_token_id(self) -> int:
        return self._eot_id

    @property
    def pad_token_id(self) -> int:
        return self._eot_id  # Same convention as GPT-2: use EOS as PAD

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)