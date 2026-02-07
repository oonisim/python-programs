"""Training script for a small Language Model (mini LLM) on WikiText-2.

Dataset: WikiText-2 (wikitext-2-raw-v1)
- Wikipedia articles
- ~2M tokens, ~12MB
- Good for learning and small models

Model: Decoder-only Transformer (GPT-style)
- Uses causal (masked) self-attention
- Predicts next token given previous tokens

Requirements:
    pip install datasets transformers torch

Usage:
    python -m scratch.trainer_wikitext_lm

    # With custom parameters:
    python -m scratch.trainer_wikitext_lm --epochs 10 --batch_size 32

    # Quick test:
    python -m scratch.trainer_wikitext_lm --max_samples 1000

    # Resume from checkpoint:
    python -m scratch.trainer_wikitext_lm --resume checkpoints/lm_epoch_5.pt
"""
import argparse
import os
from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# HuggingFace
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Local imports
try:
    from .constant import TYPE_FLOAT
    from .common import (
        PositionalEncoding,
        InputEmbedding,
        MultiHeadAttention,
        PositionwiseFeedForward,
    )
except ImportError:
    from constant import TYPE_FLOAT
    from common import (
        PositionalEncoding,
        InputEmbedding,
        MultiHeadAttention,
        PositionwiseFeedForward,
    )


# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
class Config:
    """Training configuration"""
    # Model (small GPT-style)
    d_model: int = 256              # Model dimension
    num_heads: int = 4              # Attention heads
    num_layers: int = 4             # Decoder layers
    d_ff: int = 512                 # Feed-forward dimension
    max_seq_len: int = 256          # Context window
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    tokenizer_name: str = "gpt2"
    max_samples: int = None         # None = use all

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------
# Decoder-Only Language Model (GPT-style)
# --------------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """Single decoder layer with causal self-attention."""

    def __init__(
        self,
        i_layer: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        dtype=TYPE_FLOAT
    ):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model, dtype=dtype)
        self.causal_attention = MultiHeadAttention(
            i_layer=i_layer,
            num_heads=num_heads,
            d_model=d_model,
            dtype=dtype,
            do_mask=True,  # Causal masking
            max_time_steps=max_seq_len,
            bias=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.layer_norm2 = nn.LayerNorm(d_model, dtype=dtype)
        self.feedforward = PositionwiseFeedForward(
            i_layer=i_layer,
            d_model=d_model,
            d_ff=d_ff,
            dtype=dtype,
            bias=True
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm architecture
        # Self-attention with residual
        normed = self.layer_norm1(x)
        x = x + self.dropout1(self.causal_attention(q=normed, k=normed, v=normed))

        # Feed-forward with residual
        x = x + self.dropout2(self.feedforward(self.layer_norm2(x)))

        return x


class LanguageModel(nn.Module):
    """Decoder-only Transformer for language modeling (GPT-style).

    Architecture:
        Input tokens -> Embedding + Positional Encoding -> N x DecoderLayers -> Output projection

    Usage:
        model = LanguageModel(vocab_size=50257, ...)

        # Training: get log probabilities
        log_probs = model.forward(input_ids)  # (B, T, V)
        loss = criterion(log_probs.view(-1, V), target.view(-1))

        # Inference: generate text
        with model:
            output = model.generate(prompt_ids, max_length=100)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        dtype=TYPE_FLOAT
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = InputEmbedding(
            vocabulary_size=vocab_size,
            d_model=d_model,
            dtype=dtype
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            max_time_steps=max_seq_len,
            d_model=d_model,
            dtype=dtype
        )
        self.dropout = nn.Dropout(dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                i_layer=i,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                dtype=dtype
            )
            for i in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model, dtype=dtype)

        # Output projection (tied weights with embedding optional)
        self.output_projection = nn.Linear(d_model, vocab_size, dtype=dtype)

        # Special tokens for generation
        self.start_token: Optional[int] = None
        self.end_token: Optional[int] = None

    def __enter__(self):
        """Context manager: set eval mode."""
        self.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: restore train mode."""
        self.train()
        return False

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for training.

        Args:
            x: Input token IDs of shape (B, T)

        Returns:
            Log probabilities of shape (B, T, V)
        """
        assert x.ndim == 2, f"Expected (B, T), got {x.shape}"
        B, T = x.shape

        # Embedding + positional encoding
        x = self.embedding(x)  # (B, T, D)
        x = self.dropout(x + self.positional_encoding(x))

        # Decoder layers
        for layer in self.layers:
            x = layer(x)

        # Final norm and projection
        x = self.final_norm(x)
        logits = self.output_projection(x)  # (B, T, V)

        # Return log probabilities
        return torch.log_softmax(logits, dim=-1)

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
            prompt: Starting token IDs of shape (B, T) or (T,)
            max_length: Maximum total length to generate
            temperature: Sampling temperature (1.0 = normal, <1 = more deterministic)
            top_k: Keep only top k tokens for sampling
            top_p: Keep tokens with cumulative probability <= top_p (nucleus sampling)

        Returns:
            Generated token IDs of shape (B, max_length)
        """
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0)

        B = prompt.shape[0]
        device = prompt.device
        generated = prompt.clone()

        for _ in range(max_length - prompt.shape[1]):
            # Get predictions for next token
            # Only use last max_seq_len tokens if sequence is too long
            input_seq = generated[:, -self.max_seq_len:]
            log_probs = self.forward(input_seq)
            next_token_logits = log_probs[:, -1, :] / temperature  # (B, V)

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                for b in range(B):
                    indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                    next_token_logits[b, indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token generated (for all sequences in batch)
            if self.end_token is not None and (next_token == self.end_token).all():
                break

        return generated


# --------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------
class WikiTextDataset(Dataset):
    """Dataset for WikiText language modeling."""

    def __init__(self, tokens: Tensor, seq_len: int):
        """
        Args:
            tokens: All tokens as a single 1D tensor
            seq_len: Sequence length for each sample
        """
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        # Number of complete sequences we can create
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        # Input: tokens[idx : idx + seq_len]
        # Target: tokens[idx + 1 : idx + seq_len + 1] (shifted by 1)
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def tokenize_dataset(dataset, tokenizer, max_samples=None):
    """Tokenize entire dataset into a single token sequence."""
    all_tokens = []

    data = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    for item in data:
        text = item["text"]
        if text.strip():  # Skip empty lines
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

    return torch.tensor(all_tokens, dtype=torch.long)


# --------------------------------------------------------------------------------
# Training Functions
# --------------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: Config,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(config.device)
        y = y.to(config.device)

        optimizer.zero_grad()

        # Forward pass
        log_probs = model(x)  # (B, T, V)

        # Calculate loss
        loss = criterion(log_probs.view(-1, log_probs.size(-1)), y.view(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item()

        # Progress logging
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            avg_loss = total_loss / (batch_idx + 1)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{num_batches} | "
                  f"Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: Config
) -> tuple:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(config.device)
            y = y.to(config.device)

            log_probs = model(x)
            loss = criterion(log_probs.view(-1, log_probs.size(-1)), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def generate_sample(
    model: nn.Module,
    tokenizer,
    prompt: str,
    config: Config,
    max_new_tokens: int = 50
) -> str:
    """Generate text from a prompt."""
    model.eval()

    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)

    # Generate
    with model:
        output_ids = model.generate(
            prompt_ids,
            max_length=prompt_ids.shape[1] + max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: Config,
    path: str
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": vars(config),
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None):
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


# --------------------------------------------------------------------------------
# Main Training Loop
# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Language Model on WikiText-2")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (None=all)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Configuration
    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.max_samples = args.max_samples

    print("=" * 60)
    print("Language Model Training (WikiText-2)")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Model: d_model={config.d_model}, heads={config.num_heads}, layers={config.num_layers}")
    print(f"Training: epochs={config.epochs}, batch_size={config.batch_size}, lr={config.learning_rate}")
    print()

    # --------------------------------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(config.tokenizer_name)

    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  EOS token: {tokenizer.eos_token_id}")

    # --------------------------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------------------------
    print("\nLoading dataset...")
    print(f"  Source: {config.dataset_name}/{config.dataset_config}")

    dataset = load_dataset(config.dataset_name, config.dataset_config)

    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Val samples: {len(dataset['validation'])}")
    print(f"  Test samples: {len(dataset['test'])}")

    # Tokenize datasets
    print("\nTokenizing datasets...")
    train_tokens = tokenize_dataset(dataset["train"], tokenizer, config.max_samples)
    val_tokens = tokenize_dataset(dataset["validation"], tokenizer)

    print(f"  Train tokens: {len(train_tokens):,}")
    print(f"  Val tokens: {len(val_tokens):,}")

    # Create datasets
    train_dataset = WikiTextDataset(train_tokens, config.max_seq_len)
    val_dataset = WikiTextDataset(val_tokens, config.max_seq_len)

    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------------------------------
    # Create model
    # --------------------------------------------------------------------------------
    print("\nCreating model...")
    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    model = model.to(config.device)

    # Set special tokens
    model.end_token = tokenizer.eos_token_id

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # --------------------------------------------------------------------------------
    # Optimizer and loss
    # --------------------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # NLLLoss since model returns log_probs
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1

    # --------------------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config, epoch)

        # Validate
        val_loss, val_perplexity = evaluate(model, val_loader, criterion, config)

        # Update scheduler
        scheduler.step()

        train_perplexity = torch.exp(torch.tensor(train_loss)).item()
        print(f"  Train Loss: {train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
        print(f"  Val Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, config,
                f"{config.checkpoint_dir}/lm_best_model.pt"
            )

        # Save periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, config,
                f"{config.checkpoint_dir}/lm_epoch_{epoch}.pt"
            )

        # Sample generation
        prompts = [
            "The meaning of life is",
            "In the beginning",
            "Scientists discovered that",
        ]
        print("\n  Sample generations:")
        for prompt in prompts:
            generated = generate_sample(model, tokenizer, prompt, config, max_new_tokens=30)
            print(f"    Prompt: \"{prompt}\"")
            print(f"    Generated: \"{generated}\"")
            print()

    # --------------------------------------------------------------------------------
    # Save final model
    # --------------------------------------------------------------------------------
    final_path = f"{config.checkpoint_dir}/lm_final_model.pt"
    save_checkpoint(model, optimizer, config.epochs, val_loss, config, final_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best perplexity: {torch.exp(torch.tensor(best_val_loss)).item():.2f}")
    print(f"Final model saved: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
