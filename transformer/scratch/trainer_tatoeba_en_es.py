"""Training script for English to Spanish Transformer translator.

Dataset: Tatoeba (Helsinki-NLP/tatoeba_mt)
- Simple community-contributed sentence pairs
- ~100K English-Spanish pairs
- Good for learning and small models

Requirements:
    pip install datasets transformers torch

Usage:
    python -m scratch.trainer_tatoeba_en_es

    # With custom parameters:
    python -m scratch.trainer_tatoeba_en_es --epochs 10 --batch_size 32

    # Resume from checkpoint:
    python -m scratch.trainer_tatoeba_en_es --resume checkpoints/epoch_5.pt
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# HuggingFace
from datasets import load_dataset
from transformers import AutoTokenizer

# Local imports - adjust if running as script vs module
try:
    from .model import Transformer
    from .constant import (
        DIM_MODEL, NUM_HEADS, NUM_LAYERS, DIM_PWFF_HIDDEN,
        MAX_TIME_STEPS, DROPOUT_RATIO, TYPE_FLOAT
    )
except ImportError:
    from model import Transformer
    from constant import (
        DIM_MODEL, NUM_HEADS, NUM_LAYERS, DIM_PWFF_HIDDEN,
        MAX_TIME_STEPS, DROPOUT_RATIO, TYPE_FLOAT
    )


# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
class Config:
    """Training configuration"""
    # Model (smaller for faster training)
    d_model: int = 256              # Model dimension (original: 512)
    num_heads: int = 4              # Attention heads (original: 8)
    num_layers: int = 4             # Encoder/Decoder layers (original: 6)
    d_ff: int = 512                 # Feed-forward dimension (original: 2048)
    max_seq_len: int = 128          # Max sequence length
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1

    # Data
    dataset_name: str = "Helsinki-NLP/tatoeba_mt"
    source_lang: str = "eng"
    target_lang: str = "spa"
    tokenizer_name: str = "Helsinki-NLP/opus-mt-en-es"
    max_samples: int = None         # None = use all, set to e.g. 10000 for quick test

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------
class TranslationDataset(Dataset):
    """Dataset for EN-ES translation pairs."""

    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = 128,
        source_lang: str = "en",
        target_lang: str = "es"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get source and target text
        source_text = item["sourceString"]
        target_text = item["targetString"]

        # Tokenize source (encoder input)
        source = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target (decoder input/output)
        target = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "source_ids": source["input_ids"].squeeze(0),
            "target_ids": target["input_ids"].squeeze(0),
            "source_text": source_text,
            "target_text": target_text,
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "source_ids": torch.stack([item["source_ids"] for item in batch]),
        "target_ids": torch.stack([item["target_ids"] for item in batch]),
    }


# --------------------------------------------------------------------------------
# Training Functions
# --------------------------------------------------------------------------------
def create_target_mask(target_ids: Tensor, pad_token_id: int) -> Tensor:
    """Create mask to ignore padding tokens in loss calculation."""
    return target_ids != pad_token_id


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: Config,
    epoch: int,
    pad_token_id: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        source_ids = batch["source_ids"].to(config.device)
        target_ids = batch["target_ids"].to(config.device)

        # Decoder input: shift right (prepend start token, remove last)
        # Target output: original (what we want to predict)
        decoder_input = target_ids[:, :-1]
        decoder_target = target_ids[:, 1:]

        # Forward pass
        optimizer.zero_grad()
        log_probs = model.forward(x=source_ids, y=decoder_input)

        # Reshape for loss calculation
        # log_probs: (B, T, V) -> (B*T, V)
        # target: (B, T) -> (B*T,)
        log_probs_flat = log_probs.reshape(-1, log_probs.size(-1))
        target_flat = decoder_target.reshape(-1)

        # Calculate loss (ignore padding)
        loss = criterion(log_probs_flat, target_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item()

        # Progress logging
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{num_batches} | Loss: {avg_loss:.4f}")

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: Config,
    pad_token_id: int
) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            source_ids = batch["source_ids"].to(config.device)
            target_ids = batch["target_ids"].to(config.device)

            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]

            log_probs = model.forward(x=source_ids, y=decoder_input)

            log_probs_flat = log_probs.reshape(-1, log_probs.size(-1))
            target_flat = decoder_target.reshape(-1)

            loss = criterion(log_probs_flat, target_flat)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def translate_sample(
    model: nn.Module,
    tokenizer,
    source_text: str,
    config: Config
) -> str:
    """Translate a single sentence for demonstration."""
    model.eval()

    # Tokenize source
    source = tokenizer(
        source_text,
        max_length=config.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    source_ids = source["input_ids"].to(config.device)

    # Generate translation
    with model:
        output_ids = model(
            source_ids,
            start_token=tokenizer.pad_token_id or 0,
            end_token=tokenizer.eos_token_id or 1,
            max_length=config.max_seq_len
        )

    # Decode
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translation


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
    parser = argparse.ArgumentParser(description="Train EN-ES Transformer translator")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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
    print("Transformer EN→ES Translator Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Model: d_model={config.d_model}, heads={config.num_heads}, layers={config.num_layers}")
    print(f"Training: epochs={config.epochs}, batch_size={config.batch_size}, lr={config.learning_rate}")
    print()

    # --------------------------------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  PAD token: {tokenizer.pad_token_id}")
    print(f"  EOS token: {tokenizer.eos_token_id}")

    # --------------------------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------------------------
    print("\nLoading dataset...")
    print(f"  Source: {config.dataset_name}")

    # Load Tatoeba EN-ES
    dataset = load_dataset(
        config.dataset_name,
        lang1=config.source_lang,
        lang2=config.target_lang,
        split="train"
    )

    # Limit samples if specified
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    print(f"  Total samples: {len(dataset)}")

    # Split into train/val
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    val_data = split["test"]
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")

    # Create datasets
    train_dataset = TranslationDataset(
        train_data, tokenizer, config.max_seq_len,
        config.source_lang, config.target_lang
    )
    val_dataset = TranslationDataset(
        val_data, tokenizer, config.max_seq_len,
        config.source_lang, config.target_lang
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --------------------------------------------------------------------------------
    # Create model
    # --------------------------------------------------------------------------------
    print("\nCreating model...")

    # Update constants for smaller model
    import scratch.constant as const
    const.DIM_MODEL = config.d_model
    const.NUM_HEADS = config.num_heads
    const.NUM_LAYERS = config.num_layers
    const.DIM_PWFF_HIDDEN = config.d_ff
    const.MAX_TIME_STEPS = config.max_seq_len
    const.NUM_ENCODER_TOKENS = vocab_size
    const.NUM_DECODER_TOKENS = vocab_size
    const.NUM_CLASSES = vocab_size

    # Reload model module to pick up new constants
    import importlib
    import model as model_module
    importlib.reload(model_module)

    model = model_module.Transformer()
    model = model.to(config.device)

    # Set special tokens
    model.start_token = tokenizer.pad_token_id or 0
    model.end_token = tokenizer.eos_token_id or 1

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # --------------------------------------------------------------------------------
    # Optimizer and loss
    # --------------------------------------------------------------------------------
    optimizer = Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # CrossEntropyLoss with label smoothing (expects logits, not log_probs)
    # Since our model returns log_probs, we use NLLLoss
    criterion = nn.NLLLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=config.label_smoothing
    )

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
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            config, epoch, tokenizer.pad_token_id
        )

        # Validate
        val_loss = evaluate(model, val_loader, criterion, config, tokenizer.pad_token_id)

        # Update scheduler
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, config,
                f"{config.checkpoint_dir}/best_model.pt"
            )

        # Save periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, config,
                f"{config.checkpoint_dir}/epoch_{epoch}.pt"
            )

        # Sample translation
        sample_sentences = [
            "Hello, how are you?",
            "I love programming.",
            "The weather is nice today.",
        ]
        print("\n  Sample translations:")
        for sentence in sample_sentences:
            translation = translate_sample(model, tokenizer, sentence, config)
            print(f"    EN: {sentence}")
            print(f"    ES: {translation}")
            print()

    # --------------------------------------------------------------------------------
    # Save final model
    # --------------------------------------------------------------------------------
    final_path = f"{config.checkpoint_dir}/final_model.pt"
    save_checkpoint(model, optimizer, config.epochs, val_loss, config, final_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
