"""Language Model Training Director.

This module orchestrates the language model training pipeline using the
Director pattern (GoF). It coordinates the components without knowing
their internal details.

Architecture
------------
The training system follows clean separation of concerns:

    common.py          # Transformer components (MHA, FFN, etc.)
    encoder.py         # Encoder stack
    decoder.py         # Decoder stack (memory optional for LM)
    model.py           # Encoder-Decoder Transformer
    lm.py              # LanguageModel (decoder-only, GPT-style)
    loader.py          # Data loading with tokenizer dependency injection
    trainer.py         # Trainer + LanguageModelTrainer
    train_lm.py        # Training Director (this file)
    app.py             # TransformerAPI wrapper
    utility.py         # File/checkpoint utilities

Pipeline Flow
-------------
The Director orchestrates the following pipeline:

    CLI args -> Director -> Tokenizer (GPT2)
                         -> DataLoaderFactory (loader.py)
                         -> LanguageModel (lm.py)
                         -> LanguageModelTrainer (trainer.py)
                         -> Training execution
                         -> Model saving

Director Steps:
    1. _build_tokenizer()    - Create GPT-2 BPE tokenizer
    2. _build_data_loaders() - Ingest dataset, tokenize, create DataLoaders
    3. _build_model()        - Create LanguageModel (decoder-only Transformer)
    4. _build_trainer()      - Create optimizer, scheduler, LanguageModelTrainer
    5. _execute_training()   - Run training loop with checkpointing

Usage
-----
Basic training:
    python train_lm.py --dataset wikitext --epochs 10

Quick test with fewer samples:
    python train_lm.py --dataset wikitext --max_samples 1000

Resume from checkpoint:
    python train_lm.py --resume

Show this documentation:
    python train_lm.py --info

Available Datasets
------------------
    - wikitext: WikiText-2 (~2M tokens) - Wikipedia articles
    - wikitext-103: WikiText-103 (~100M tokens) - Larger Wikipedia
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Tokenizer

from model.lm import LanguageModel
from training.loader import LanguageModelDataLoaderFactory, DataLoaderConfig
from training.trainer import LanguageModelTrainer, TrainerConfig
from training.trainer_early_stopping import EarlyStoppingCallback
from training.trainer_gradient_monitor import GradientMonitorCallback


# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Language model architecture configuration."""
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    log_interval: int = 100
    snapshot_interval: int = 0
    keep_last_n_snapshots: int = 3
    delete_snapshots_after_training: bool = True
    # Callback options
    enable_gradient_monitor: bool = False
    gradient_monitor_interval: int = 0  # Monitor every N steps (0 = at snapshots only)
    enable_early_stopping: bool = False
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.001
    early_stop_restore_best: bool = True
    early_stop_overfit_patience: int = 0  # Stop if val-train gap grows for N epochs (0 = disabled)
    early_stop_overfit_min_delta: float = 0.01  # Minimum gap increase to count

    # Weight update monitoring (frozen weight detection)
    enable_weight_monitor: bool = True  # Enabled by default (adds ~1-2% overhead)
    weight_monitor_interval: int = 100
    weight_monitor_sample_size: int = 1024
    monitor_topk: int = 5
    vanishing_grad_threshold: float = 1e-7
    exploding_grad_threshold: float = 1e2
    frozen_update_ratio_threshold: float = 1e-12
    frozen_patience_steps: int = 3


DATASET_CONFIGS = {
    "wikitext": ("wikitext", "wikitext-2-raw-v1"),
    "wikitext-103": ("wikitext", "wikitext-103-raw-v1"),
}

TOKENIZER_CONFIGS = {
    "gpt2": None,           # Uses HuggingFace GPT2Tokenizer
    "gpt4": "cl100k_base",  # tiktoken encoding for GPT-4
    "gpt4o": "o200k_base",  # tiktoken encoding for GPT-4o
}


# --------------------------------------------------------------------------------
# Tiktoken Adapter
# --------------------------------------------------------------------------------
class TiktokenAdapter:
    """Adapter for tiktoken to match the Tokenizer protocol in loader.py."""

    def __init__(self, encoding_name: str):
        import tiktoken
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


# --------------------------------------------------------------------------------
# Director
# --------------------------------------------------------------------------------
class LanguageModelTrainingDirector:
    """Director class orchestrating the LM training pipeline.

    Coordinates the training process without knowing internal details
    of the components. Each step in the pipeline is clearly defined.
    """

    def __init__(
            self,
            dataset_key: str,
            model_config: ModelConfig,
            training_config: TrainingConfig,
            max_samples: Optional[int] = None,
            resume: bool = False,
            tokenizer_name: str = "gpt2",
            checkpoint_file: Optional[str] = None,
            auto_confirm: bool = False
    ):
        """Initialize the director.

        Args:
            dataset_key: Dataset identifier (e.g., "wikitext").
            model_config: Model architecture configuration.
            training_config: Training hyperparameters.
            max_samples: Limit training samples (for quick testing).
            resume: Whether to resume from checkpoint.
            tokenizer_name: Tokenizer to use (key in TOKENIZER_CONFIGS).
            checkpoint_file: Path to specific checkpoint file to load.
            auto_confirm: Skip confirmation prompts for checkpoint loading.
        """
        self.dataset_key = dataset_key
        self.model_config = model_config
        self.training_config = training_config
        self.max_samples = max_samples
        self.resume = resume
        self.tokenizer_name = tokenizer_name
        self.checkpoint_file = checkpoint_file
        self.auto_confirm = auto_confirm

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Components (built during pipeline execution)
        self.tokenizer = None
        self.data_factory = None
        self.model = None
        self.trainer = None

    def run(self) -> dict:
        """Execute the complete training pipeline.

        Returns:
            Dictionary with training results.
        """
        if self.resume:
            self._load_run_config()
        self._print_header()
        self._build_tokenizer()
        self._build_data_loaders()
        self._build_model()
        self._build_trainer()
        if not self.resume:
            self._save_run_config()
        return self._execute_training()

    def _run_config_path(self) -> Path:
        """Path to the saved run configuration file."""
        return Path(f"lm_{self.dataset_key}") / "snapshots" / "run_config.json"

    def _save_run_config(self) -> None:
        """Save run configuration to disk for safe resume."""
        config_path = self._run_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "dataset_key": self.dataset_key,
            "tokenizer_name": self.tokenizer_name,
            "max_samples": self.max_samples,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
        }
        config_path.write_text(json.dumps(config, indent=2))
        print(f"  Run config saved: {config_path}")

    def _load_run_config(self) -> None:
        """Load saved run configuration, overriding CLI args."""
        config_path = self._run_config_path()
        if not config_path.exists():
            print(f"WARNING: No saved run config at {config_path}, using CLI args.")
            return
        config = json.loads(config_path.read_text())
        self.tokenizer_name = config["tokenizer_name"]
        self.max_samples = config["max_samples"]
        self.model_config = ModelConfig(**config["model_config"])
        self.training_config = TrainingConfig(**config["training_config"])
        print(f"Loaded run config from {config_path}, ignoring CLI args.")

    def _print_header(self) -> None:
        """Print training session header."""
        print("=" * 60)
        print("Language Model Training")
        print("=" * 60)
        print(f"Dataset: {self.dataset_key}")
        print(f"Device: {self.device}")
        print(f"Model: d={self.model_config.d_model}, "
              f"h={self.model_config.num_heads}, "
              f"L={self.model_config.num_layers}")
        print(f"Training: epochs={self.training_config.epochs}, "
              f"batch={self.training_config.batch_size}, "
              f"lr={self.training_config.learning_rate}")
        print()

    def _build_tokenizer(self) -> None:
        """Step 1: Create tokenizer."""
        print(f"Building tokenizer ({self.tokenizer_name})...")
        encoding_name = TOKENIZER_CONFIGS[self.tokenizer_name]

        if encoding_name is None:
            # HuggingFace GPT-2 tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # tiktoken-based tokenizer
            self.tokenizer = TiktokenAdapter(encoding_name)

        print(f"  Vocabulary size: {self.tokenizer.vocab_size}")

    def _build_data_loaders(self) -> None:
        """Step 2: Create data loaders."""
        print("\nBuilding data loaders...")

        dataset_name, dataset_config = DATASET_CONFIGS[self.dataset_key]

        loader_config = DataLoaderConfig(
            seq_len=self.model_config.max_seq_len,
            batch_size=self.training_config.batch_size,
            max_train_samples=self.max_samples
        )

        self.data_factory = LanguageModelDataLoaderFactory(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            tokenizer=self.tokenizer,
            config=loader_config
        )

        # Trigger data loading to show stats
        _ = self.data_factory.get_train_loader()
        _ = self.data_factory.get_val_loader()

        stats = self.data_factory.get_stats()
        print(f"  Train tokens: {stats.get('train_tokens', 'N/A'):,}")
        print(f"  Val tokens: {stats.get('val_tokens', 'N/A'):,}")
        print(f"  Train sequences: {stats.get('train_sequences', 'N/A'):,}")

    def _build_model(self) -> None:
        """Step 3: Create model."""
        print("\nBuilding model...")

        self.model = LanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.model_config.d_model,
            num_heads=self.model_config.num_heads,
            num_layers=self.model_config.num_layers,
            d_ff=self.model_config.d_ff,
            max_seq_len=self.model_config.max_seq_len,
            dropout=self.model_config.dropout
        )

        self.model.end_token = self.tokenizer.eos_token_id

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {num_params:,}")

    def _build_callbacks(self) -> list:
        """Build training callbacks based on config.

        Returns:
            List of callback instances.
        """
        callbacks = []

        # Early stopping callback
        if self.training_config.enable_early_stopping:
            early_stop = EarlyStoppingCallback(
                patience=self.training_config.early_stop_patience,
                min_delta=self.training_config.early_stop_min_delta,
                restore_best=self.training_config.early_stop_restore_best,
                overfit_patience=self.training_config.early_stop_overfit_patience,
                overfit_min_delta=self.training_config.early_stop_overfit_min_delta
            )
            callbacks.append(early_stop)
            overfit_msg = f", overfit_patience={self.training_config.early_stop_overfit_patience}" \
                if self.training_config.early_stop_overfit_patience > 0 else ""
            print(f"  Early stopping enabled (patience={self.training_config.early_stop_patience}{overfit_msg})")

        # Gradient monitor callback
        if self.training_config.enable_gradient_monitor:
            gradient_monitor = GradientMonitorCallback(
                monitor_at_snapshots=self.training_config.snapshot_interval > 0,
                monitor_interval=self.training_config.gradient_monitor_interval,
                monitor_at_epochs=False
            )
            callbacks.append(gradient_monitor)
            print(f"  Gradient monitoring enabled")

        return callbacks

    def _build_trainer(self) -> None:
        """Step 4: Create trainer with optimizer and scheduler."""
        print("\nBuilding trainer...")

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.training_config.weight_decay
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.training_config.epochs
        )

        # Model returns log-probabilities. Do NOT use CrossEntropyLoss as
        # it expects logits and internally applies log-softmax + NLLLoss.
        criterion = nn.NLLLoss(ignore_index=self.tokenizer.pad_token_id)

        trainer_config = TrainerConfig(
            model_name=f"lm_{self.dataset_key}",
            gradient_clip=self.training_config.gradient_clip,
            log_interval=self.training_config.log_interval,
            snapshot_interval=self.training_config.snapshot_interval,
            snapshot_per_epoch=True,
            keep_last_n_snapshots=self.training_config.keep_last_n_snapshots,
            delete_snapshots_after_training=self.training_config.delete_snapshots_after_training,
            enable_weight_monitor=self.training_config.enable_weight_monitor,
            weight_monitor_interval=self.training_config.weight_monitor_interval,
            weight_monitor_sample_size=self.training_config.weight_monitor_sample_size,
            monitor_topk=self.training_config.monitor_topk,
            vanishing_grad_threshold=self.training_config.vanishing_grad_threshold,
            exploding_grad_threshold=self.training_config.exploding_grad_threshold,
            frozen_update_ratio_threshold=self.training_config.frozen_update_ratio_threshold,
            frozen_patience_steps=self.training_config.frozen_patience_steps,
        )

        # Build callbacks
        callbacks = self._build_callbacks()

        self.trainer = LanguageModelTrainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            config=trainer_config,
            device=self.device,
            scheduler=scheduler,
            callbacks=callbacks
        )

    def _find_latest_model(self) -> Optional[Path]:
        """Find the most recent model file in models directory.

        Returns:
            Path to latest model, or None if none exist.
        """
        models_dir = Path(f"lm_{self.dataset_key}") / "models"
        if not models_dir.exists():
            return None

        model_files = sorted(models_dir.glob("model_*.pt"),
                           key=lambda p: p.stat().st_mtime,
                           reverse=True)
        return model_files[0] if model_files else None

    def _confirm_checkpoint(self, checkpoint_path: Path, source: str) -> bool:
        """Ask user to confirm loading a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            source: Description of where checkpoint was found (e.g., "snapshots", "models").

        Returns:
            True if user confirms, False otherwise.
        """
        print(f"\nFound checkpoint in {source}: {checkpoint_path}")

        # Try to load and show info
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if "epoch" in ckpt:
                print(f"  Epoch: {ckpt['epoch']}")
            if "training_history" in ckpt:
                epochs_trained = len(ckpt["training_history"])
                print(f"  Epochs trained: {epochs_trained}")
                if ckpt["training_history"]:
                    last = ckpt["training_history"][-1]
                    print(f"  Last train loss: {last.get('train_loss', 'N/A'):.4f}")
                    print(f"  Last val loss: {last.get('val_loss', 'N/A'):.4f}")
            if "best_val_loss" in ckpt:
                print(f"  Best val loss: {ckpt['best_val_loss']:.4f}")
        except Exception as e:
            print(f"  (Could not read checkpoint info: {e})")

        # Auto-confirm if flag is set
        if self.auto_confirm:
            print("Auto-confirming (--yes flag set)")
            return True

        response = input("\nLoad this checkpoint? [Y/n]: ").strip().lower()
        return response in ["", "y", "yes"]

    def _load_checkpoint_file(self, checkpoint_path: Path) -> int:
        """Load a checkpoint file and return starting epoch.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Starting epoch number.
        """
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Determine starting epoch
        start_epoch = 0
        if "epoch" in checkpoint:
            # Snapshot format: has epoch field
            start_epoch = checkpoint["epoch"] + 1
            print(f"Loaded snapshot from epoch {checkpoint['epoch']}")
        elif "training_history" in checkpoint:
            # Final model format: has training_history
            start_epoch = len(checkpoint["training_history"])
            print(f"Loaded model trained for {start_epoch} epochs")

        # Try to restore optimizer and scheduler if available
        if "optimizer_state_dict" in checkpoint:
            self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Restored optimizer state")

        if "scheduler_state_dict" in checkpoint and self.trainer.scheduler is not None:
            self.trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("Restored scheduler state")

        return start_epoch

    def _execute_training(self) -> dict:
        """Step 5: Execute training loop.

        Returns:
            Dictionary with training results.
        """
        train_loader = self.data_factory.get_train_loader()
        val_loader = self.data_factory.get_val_loader()

        # Handle resume
        start_epoch = 0
        if self.resume or self.checkpoint_file:
            checkpoint_path = None

            # If specific checkpoint file is provided, use it
            if self.checkpoint_file:
                checkpoint_path = Path(self.checkpoint_file)
                if not checkpoint_path.exists():
                    print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
                    print("Starting fresh training instead.")
                else:
                    if self._confirm_checkpoint(checkpoint_path, "specified file"):
                        start_epoch = self._load_checkpoint_file(checkpoint_path)
                    else:
                        print("Checkpoint loading cancelled. Starting fresh.")

            # Otherwise, search for latest checkpoint
            elif self.resume:
                # First, try to find snapshot
                latest_snapshot = self.trainer.find_latest_snapshot()
                if latest_snapshot:
                    if self._confirm_checkpoint(latest_snapshot, "snapshots"):
                        checkpoint = self.trainer.load_snapshot(latest_snapshot.name)
                        start_epoch = checkpoint["epoch"] + 1
                    else:
                        print("Snapshot loading cancelled.")
                        # Try models directory instead
                        latest_model = self._find_latest_model()
                        if latest_model:
                            if self._confirm_checkpoint(latest_model, "models"):
                                start_epoch = self._load_checkpoint_file(latest_model)
                            else:
                                print("Starting fresh training.")
                        else:
                            print("No models found. Starting fresh.")
                else:
                    # No snapshot, try models directory
                    print("No snapshot found in snapshots directory.")
                    latest_model = self._find_latest_model()
                    if latest_model:
                        if self._confirm_checkpoint(latest_model, "models"):
                            start_epoch = self._load_checkpoint_file(latest_model)
                        else:
                            print("Starting fresh training.")
                    else:
                        print("No checkpoints found. Starting fresh.")

        print("\n" + "=" * 60)
        print("Training...")
        print("=" * 60)

        # Training loop
        result = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.training_config.epochs,
            start_epoch=start_epoch
        )

        # Print summary
        self._print_summary(result)

        return result

    def _print_summary(self, result: dict) -> None:
        """Print training summary."""
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best validation loss: {result['best_val_loss']:.4f}")

        perplexity = torch.exp(torch.tensor(result['best_val_loss'])).item()
        print(f"Best perplexity: {perplexity:.2f}")
        print(f"Model saved: {result['model_path']}")


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
HELP_TEXT = """
Language Model Training Script
==============================

This script trains a decoder-only Transformer (GPT-style) language model.
The model learns to predict the next token given previous tokens using
causal self-attention.

WHAT IS A LANGUAGE MODEL?
-------------------------
A language model predicts the probability of the next word given previous words.
For example, given "The cat sat on the", it predicts "mat" has high probability.

This is the architecture behind GPT-2, GPT-3, GPT-4, LLaMA, and Mistral.

HOW IT WORKS
------------
1. Text is tokenized into subword tokens using GPT-2's BPE tokenizer
2. The model processes tokens through multiple decoder layers
3. Each layer uses causal (masked) attention - tokens can only see past tokens
4. The model outputs probability distribution over vocabulary for next token
5. Training minimizes negative log-likelihood (cross-entropy) loss

QUICK START
-----------
# Train with default settings (small model, WikiText-2)
python train_lm.py

# Quick test with fewer samples
python train_lm.py --max_samples 1000

# Train larger model for more epochs
python train_lm.py --d_model 512 --num_layers 6 --epochs 50

# Resume interrupted training
python train_lm.py --resume

OUTPUT
------
After training, the model is saved to:
    lm_<dataset>/models/model_<timestamp>.pt

Use it for text generation:
    from lm import LanguageModel
    model = LanguageModel(vocab_size=50257)
    model.load_state_dict(torch.load("path/to/model.pt")["model_state_dict"])

    with model:
        output = model.generate(prompt_ids, max_length=100)

PERPLEXITY
----------
Perplexity measures how "surprised" the model is by test data.
- Lower is better
- Perplexity of 100 = model is as confused as picking from 100 random tokens
- Good LMs achieve perplexity < 30 on WikiText-2
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a GPT-style language model on text datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use --info for detailed documentation, --help for argument help."
    )

    # --------------------------------------------------------------------------------
    # Data Arguments
    # --------------------------------------------------------------------------------
    data_group = parser.add_argument_group(
        "Data",
        "Dataset selection and preprocessing options."
    )
    data_group.add_argument(
        "--dataset", type=str, default="wikitext",
        choices=list(DATASET_CONFIGS.keys()),
        help=(
            "Dataset to train on. "
            "'wikitext' uses WikiText-2 (~2M tokens, good for testing). "
            "'wikitext-103' uses WikiText-103 (~100M tokens, for serious training). "
            "Default: wikitext"
        )
    )
    data_group.add_argument(
        "--tokenizer", type=str, default="gpt2",
        choices=list(TOKENIZER_CONFIGS.keys()),
        help=(
            "Tokenizer to use. "
            "'gpt2' uses GPT-2 BPE (50257 tokens). "
            "'gpt4' uses tiktoken cl100k_base (100256 tokens). "
            "'gpt4o' uses tiktoken o200k_base (200019 tokens). "
            "Default: gpt2"
        )
    )
    data_group.add_argument(
        "--max_samples", type=int, default=None,
        metavar="N",
        help=(
            "Limit the number of training samples. Useful for quick testing "
            "or debugging. For example, --max_samples 1000 trains on only "
            "1000 text samples. Default: None (use all samples)"
        )
    )

    # --------------------------------------------------------------------------------
    # Training Arguments
    # --------------------------------------------------------------------------------
    train_group = parser.add_argument_group(
        "Training",
        "Training hyperparameters controlling the optimization process."
    )
    train_group.add_argument(
        "--epochs", type=int, default=20,
        metavar="N",
        help=(
            "Number of training epochs. One epoch = one pass through entire "
            "training dataset. More epochs generally improve performance but "
            "risk overfitting. Default: 20"
        )
    )
    train_group.add_argument(
        "--batch_size", type=int, default=32,
        metavar="N",
        help=(
            "Number of sequences per gradient update. Larger batches provide "
            "more stable gradients but require more GPU memory. Reduce if "
            "you encounter OOM errors. Default: 32"
        )
    )
    train_group.add_argument(
        "--lr", type=float, default=3e-4,
        metavar="RATE",
        help=(
            "Learning rate for AdamW optimizer. Controls step size during "
            "gradient descent. Typical range: 1e-5 to 1e-3. Higher values "
            "train faster but may be unstable. Default: 3e-4"
        )
    )
    train_group.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume training from the latest checkpoint. Useful when training "
            "is interrupted. The script automatically finds the most recent "
            "snapshot in the model directory."
        )
    )
    train_group.add_argument(
        "--checkpoint_file", type=str, default=None,
        metavar="PATH",
        help=(
            "Path to a specific checkpoint file (.pt) to load and continue training. "
            "Can be a snapshot or a final model. If not specified with --resume, "
            "the script will search for the latest checkpoint automatically."
        )
    )
    train_group.add_argument(
        "--yes", "-y", action="store_true",
        help=(
            "Auto-confirm checkpoint loading without prompting. "
            "Useful for non-interactive/background execution."
        )
    )
    train_group.add_argument(
        "--snapshot_interval", type=int, default=0,
        metavar="N",
        help=(
            "Save a snapshot every N training steps. Useful for long epochs "
            "where per-epoch snapshots are too infrequent. "
            "0 disables step-based snapshots. Default: 0"
        )
    )
    train_group.add_argument(
        "--keep_last_n_snapshots", type=int, default=3,
        metavar="N",
        help=(
            "Number of recent snapshots to keep on disk. Older snapshots are "
            "deleted automatically. 0 keeps all snapshots. Default: 3"
        )
    )
    train_group.add_argument(
        "--delete_snapshots_after_training", action="store_true",
        default=True,
        help=(
            "Delete all snapshots after training completes. The final model "
            "is saved separately. Enabled by default."
        )
    )
    train_group.add_argument(
        "--no_delete_snapshots_after_training", action="store_false",
        dest="delete_snapshots_after_training",
        help=(
            "Keep snapshots after training completes. Useful for post-hoc "
            "analysis or resuming from intermediate checkpoints."
        )
    )
    train_group.add_argument(
        "--gradient_monitor", action="store_true",
        help=(
            "Enable gradient flow monitoring to detect vanishing/exploding "
            "gradients. Monitors at snapshot intervals by default. "
            "Use --gradient_monitor_interval for custom frequency."
        )
    )
    train_group.add_argument(
        "--gradient_monitor_interval", type=int, default=0,
        metavar="N",
        help=(
            "Monitor gradient flow every N steps (in addition to snapshots). "
            "0 monitors only at snapshot intervals. Requires --gradient_monitor. "
            "Default: 0"
        )
    )
    train_group.add_argument(
        "--early_stopping", action="store_true",
        help=(
            "Enable early stopping to halt training when validation loss stops "
            "improving. Use --early_stop_patience to control patience."
        )
    )
    train_group.add_argument(
        "--early_stop_patience", type=int, default=5,
        metavar="N",
        help=(
            "Number of epochs without improvement before early stopping triggers. "
            "Requires --early_stopping. Default: 5"
        )
    )
    train_group.add_argument(
        "--early_stop_min_delta", type=float, default=0.001,
        metavar="DELTA",
        help=(
            "Minimum change in loss to qualify as improvement for early stopping. "
            "Requires --early_stopping. Default: 0.001"
        )
    )
    train_group.add_argument(
        "--no_early_stop_restore_best", action="store_false",
        dest="early_stop_restore_best",
        help=(
            "Do not restore best weights when early stopping triggers. "
            "By default, best weights are restored."
        )
    )
    train_group.add_argument(
        "--early_stop_overfit_patience", type=int, default=0,
        metavar="N",
        help=(
            "Stop training if val_loss - train_loss keeps growing for N consecutive epochs. "
            "This detects overfitting. Set to 0 to disable (default). "
            "Requires --early_stopping and validation data."
        )
    )
    train_group.add_argument(
        "--early_stop_overfit_min_delta", type=float, default=0.01,
        metavar="DELTA",
        help=(
            "Minimum increase in val-train gap to count as overfitting trend. "
            "Requires --early_stop_overfit_patience > 0. Default: 0.01"
        )
    )
    train_group.add_argument(
        "--weight_monitor", action="store_true",
        help=(
            "Enable weight update monitoring to detect frozen weights. "
            "Uses a few MB of GPU memory (sample_size × num_params). "
            "Monitors actual Δw after optimizer.step() (works correctly "
            "with AdamW, weight decay, and gradient clipping)."
        )
    )
    train_group.add_argument(
        "--weight_monitor_interval", type=int, default=100,
        metavar="N",
        help=(
            "Monitor weight updates every N steps (must be >= 1). "
            "Default: 100"
        )
    )
    train_group.add_argument(
        "--weight_monitor_sample_size", type=int, default=1024,
        metavar="SIZE",
        help=(
            "Number of sampled elements per parameter for update monitoring. "
            "Default: 1024"
        )
    )

    # --------------------------------------------------------------------------------
    # Model Architecture Arguments
    # --------------------------------------------------------------------------------
    model_group = parser.add_argument_group(
        "Model Architecture",
        "Transformer architecture hyperparameters. Larger values = more capacity "
        "but slower training and more memory."
    )
    model_group.add_argument(
        "--d_model", type=int, default=256,
        metavar="DIM",
        help=(
            "Model embedding dimension. Size of token representations. "
            "GPT-2 small uses 768, GPT-2 medium uses 1024. "
            "Must be divisible by num_heads. Default: 256"
        )
    )
    model_group.add_argument(
        "--num_heads", type=int, default=4,
        metavar="N",
        help=(
            "Number of attention heads. Each head learns different attention "
            "patterns. GPT-2 small uses 12 heads. "
            "d_model must be divisible by num_heads. Default: 4"
        )
    )
    model_group.add_argument(
        "--num_layers", type=int, default=4,
        metavar="N",
        help=(
            "Number of decoder layers (depth). Each layer has self-attention "
            "and feed-forward sublayers. GPT-2 small uses 12 layers. "
            "More layers = more capacity but slower. Default: 4"
        )
    )
    model_group.add_argument(
        "--d_ff", type=int, default=512,
        metavar="DIM",
        help=(
            "Feed-forward hidden dimension. Size of the intermediate layer "
            "in position-wise FFN. Typically 4x d_model. "
            "GPT-2 small uses 3072. Default: 512"
        )
    )
    model_group.add_argument(
        "--max_seq_len", type=int, default=256,
        metavar="LEN",
        help=(
            "Maximum sequence length (context window). How many tokens the "
            "model can see at once. Longer = more context but more memory. "
            "GPT-2 uses 1024. Default: 256"
        )
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.1,
        metavar="RATE",
        help=(
            "Dropout probability for regularization. Randomly zeros elements "
            "during training to prevent overfitting. "
            "Typical range: 0.0 to 0.3. Default: 0.1"
        )
    )

    # --------------------------------------------------------------------------------
    # Information Arguments
    # --------------------------------------------------------------------------------
    info_group = parser.add_argument_group(
        "Information",
        "Help and documentation options."
    )
    info_group.add_argument(
        "--info", action="store_true",
        help=(
            "Show detailed documentation about what this script does, "
            "how language models work, and usage examples. "
            "More comprehensive than --help."
        )
    )
    info_group.add_argument(
        "--explain", action="store_true",
        help=(
            "Show explanation of what a language model is and how to use "
            "this training script with examples."
        )
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.info:
        print(__doc__)
        sys.exit(0)

    if args.explain:
        print(HELP_TEXT)
        sys.exit(0)

    model_config = ModelConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        snapshot_interval=args.snapshot_interval,
        keep_last_n_snapshots=args.keep_last_n_snapshots,
        delete_snapshots_after_training=args.delete_snapshots_after_training,
        enable_gradient_monitor=args.gradient_monitor,
        gradient_monitor_interval=args.gradient_monitor_interval,
        enable_early_stopping=args.early_stopping,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_restore_best=args.early_stop_restore_best,
        early_stop_overfit_patience=args.early_stop_overfit_patience,
        early_stop_overfit_min_delta=args.early_stop_overfit_min_delta,
        enable_weight_monitor=args.weight_monitor,
        weight_monitor_interval=args.weight_monitor_interval,
        weight_monitor_sample_size=args.weight_monitor_sample_size,
    )

    director = LanguageModelTrainingDirector(
        dataset_key=args.dataset,
        model_config=model_config,
        training_config=training_config,
        max_samples=args.max_samples,
        resume=args.resume,
        tokenizer_name=args.tokenizer,
        checkpoint_file=args.checkpoint_file,
        auto_confirm=args.yes
    )

    director.run()


if __name__ == "__main__":
    main()
