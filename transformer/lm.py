"""Decoder-only Language Model (GPT-style).

This module implements a decoder-only Transformer for language modeling,
which is the architecture used by GPT-2, GPT-3, LLaMA, Mistral, etc.

Architecture:
    Input tokens -> Embedding + Positional Encoding -> N x DecoderLayers -> Projection

    Unlike the encoder-decoder Transformer (model.py), this uses only the decoder
    with causal self-attention (no cross-attention to encoder).

Usage:
    from scratch.lm import LanguageModel
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = LanguageModel(vocab_size=tokenizer.vocab_size)

    # Training: get log probabilities
    log_probs = model.forward(input_ids)  # (B, T, V)
    loss = criterion(log_probs.view(-1, V), target.view(-1))

    # Inference: generate text
    with model:
        output = model.generate(prompt_ids, max_length=100)
"""
from typing import Optional

import logging
import torch
from torch import Tensor, nn

from .constant import (
    TYPE_FLOAT,
    DIM_MODEL,
    DIM_PWFF_HIDDEN,
    NUM_LAYERS,
    NUM_HEADS,
    MAX_TIME_STEPS,
    DROPOUT_RATIO,
)
from .common import Projection
from .decoder import Decoder


logger = logging.getLogger(__name__)


class LanguageModel(nn.Module):
    """Decoder-only Transformer for language modeling (GPT-style).

    This model uses causal self-attention only (no encoder, no cross-attention).
    Each token can only attend to previous tokens, enabling autoregressive generation.

    Architecture:
        tokens -> Decoder(memory=None) -> Projection -> log_probs

    Example tokenizer configuration (GPT-2):
        model.end_token = tokenizer.eos_token_id  # 50256
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = DIM_MODEL,
            num_heads: int = NUM_HEADS,
            num_layers: int = NUM_LAYERS,
            d_ff: int = DIM_PWFF_HIDDEN,
            max_seq_len: int = MAX_TIME_STEPS,
            dropout: float = DROPOUT_RATIO,
            dtype: torch.dtype = TYPE_FLOAT
    ):
        """Initialize the LanguageModel.

        Args:
            vocab_size: Size of the vocabulary.
            d_model: Dimension of the model embedding vector.
            num_heads: Number of attention heads.
            num_layers: Number of decoder layers.
            d_ff: Hidden dimension of position-wise feed-forward network.
            max_seq_len: Maximum sequence length (context window).
            dropout: Dropout rate.
            dtype: Data type for weights.
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Decoder with causal self-attention only (no cross-attention)
        self.decoder = Decoder(
            vocabulary_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            dtype=dtype,
            d_ff=d_ff,
            max_time_steps=max_seq_len,
            bias=True,
            p_drop=dropout
        )

        # Final layer norm (Pre-LayerNorm architecture)
        self.final_norm = nn.LayerNorm(d_model, dtype=dtype)

        # Output projection to vocabulary
        self.projection = Projection(
            d_model=d_model,
            num_classes=vocab_size,
            dtype=dtype,
            bias=True
        )

        # Token for stopping generation
        self.end_token: Optional[int] = None
        # Detect CUDA at initialization and move model to that device
        # This makes the model ready to accept inputs on GPU by default.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __enter__(self) -> 'LanguageModel':
        """Context manager: set model to eval mode for inference."""
        self.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager: restore train mode."""
        self.train()
        return False

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for training.

        Computes log probabilities for next token prediction at each position.
        $$ \\log P(x_t | x_{<t}) $$

        Args:
            x: Input token IDs of shape (B, T).

        Returns:
            Log probabilities of shape (B, T, V).
        """
        # Check device/dtype and warn the user instead of implicitly converting.
        if x.device != self._device():
            logger.warning(
                "Input tensor device %s does not match model device %s. "
                "Do not move inputs implicitly; move them explicitly to avoid surprises.",
                x.device, self._device()
            )

        assert x.ndim == 2, f"Expected (B, T), got {x.shape}"

        # Decoder without memory (decoder-only mode)
        hidden = self.decoder(y=x, memory=None)
        hidden = self.final_norm(hidden)

        # Project to vocabulary and return log probabilities
        log_probs = self.projection(y=hidden)
        return log_probs

    @torch.no_grad()
    def generate(
            self,
            prompt: Tensor,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: Optional[int] = 50,
            top_p: Optional[float] = 0.9,
    ) -> Tensor:
        """Generate text autoregressively.

        Args:
            prompt: Starting token IDs of shape (B, T) or (T,).
            max_length: Maximum total length to generate.
            temperature: Sampling temperature (1.0 = normal, <1 = more deterministic).
            top_k: Keep only top k tokens for sampling.
            top_p: Keep tokens with cumulative probability <= top_p (nucleus sampling).

        Returns:
            Generated token IDs of shape (B, max_length).
        """
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0)

        # Warn the user about device/dtype mismatches instead of moving/casting.
        if prompt.device != self._device():
            logger.warning(
                "Prompt tensor device %s does not match model device %s. "
                "Move your inputs explicitly to the desired device.",
                prompt.device, self._device()
            )

        batch_size = prompt.shape[0]
        generated = prompt.clone()

        for _ in range(max_length - prompt.shape[1]):
            # Use only last max_seq_len tokens if sequence is too long
            input_seq = generated[:, -self.max_seq_len:]

            # Get log probabilities for next token
            log_probs = self.forward(input_seq)
            next_token_logits = log_probs[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_values = torch.topk(next_token_logits, top_k)[0][:, -1, None]
                next_token_logits[next_token_logits < top_k_values] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                next_token_logits = self._apply_top_p(next_token_logits, top_p)

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token generated (for all sequences in batch)
            if self.end_token is not None and (next_token == self.end_token).all():
                break

        return generated

    def _apply_top_p(self, logits: Tensor, top_p: float) -> Tensor:
        """Apply nucleus (top-p) sampling filter to logits.

        Args:
            logits: Logits of shape (B, V).
            top_p: Cumulative probability threshold.

        Returns:
            Filtered logits with low-probability tokens set to -inf.
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Find tokens to remove (cumulative prob exceeds top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter back to original indices
        for b in range(logits.shape[0]):
            indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
            logits[b, indices_to_remove] = float('-inf')

        return logits

    def _device(self) -> torch.device:
        """Return the device where the model parameters or buffers live.

        Falls back to CPU if no parameters/buffers are present.
        """
        # Prefer explicit self.device if set
        if hasattr(self, "device") and isinstance(self.device, torch.device):
            return self.device
        # Otherwise fall back to parameters/buffers
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")


if __name__ == "__main__":
    print(__doc__)
