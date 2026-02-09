"""Trainer module for Transformer model training.

This module provides the Trainer class that handles:
- Training loop with gradient updates
- Validation/evaluation
- Checkpoint saving and loading for crash recovery
- Model saving after training completion

Architecture:
    Trainer takes a model and data loader, managing the training lifecycle.
    Separation of concerns:
    - model.py: Model architecture only
    - trainer.py: Training logic, checkpointing, saving/loading
    - app.py: Inference/application layer

Directory Structure:
    {base_dir}/{model_name}/
    ├── snapshots/    # Training checkpoints (deleted after training)
    │   └── snapshot_epoch_0005_step_001000_20240115_143052.pt
    └── models/       # Completed models
        └── model_20240115_150030.pt

Usage:
    from scratch.trainer import Trainer
    from scratch.model import Transformer

    model = Transformer()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=torch.nn.NLLLoss(),
        model_name="my_translator"
    )

    # Train
    trainer.train(train_loader, val_loader, num_epochs=10)

    # Resume from checkpoint
    trainer.load_snapshot("snapshot_epoch_0005_step_001000_20240115_143052.pt")
    trainer.train(train_loader, val_loader, num_epochs=10)
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    List, Dict, Any, Optional
)

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from utility import (
    ensure_directory_exists,
    build_snapshot_filename,
    build_model_filename,
    resolve_file_path,
    find_latest_file,
    delete_files_by_pattern,
    cleanup_old_files,
)


@dataclass
class TrainerConfig:
    """Configuration for Trainer.

    Attributes:
        model_name: Name for this training run, used as root directory.
        base_dir: Base directory for all output files.
        gradient_clip: Maximum gradient norm for clipping (None to disable).
        log_interval: Log training progress every N steps.
        snapshot_interval: Save snapshot every N steps (0 to disable step snapshots).
        snapshot_per_epoch: Save snapshot at the end of each epoch.
        keep_last_n_snapshots: Number of recent snapshots to keep (0 to keep all).
        delete_snapshots_after_training: Delete all snapshots after training completes.
    """
    model_name: str = "transformer"
    base_dir: str = "."
    gradient_clip: Optional[float] = 1.0
    log_interval: int = 100
    snapshot_interval: int = 0
    snapshot_per_epoch: bool = True
    keep_last_n_snapshots: int = 5
    delete_snapshots_after_training: bool = True


class Trainer:
    """Trainer class for model training with checkpoint management.

    Handles the training loop, validation, checkpointing, and model saving.
    Designed for separation of concerns - model.py handles architecture,
    trainer.py handles training lifecycle.

    Attributes:
        model: The neural network model to train.
        optimizer: Optimizer for gradient updates.
        criterion: Loss function.
        config: Training configuration.
        device: Device to run training on (cuda/cpu).
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            config: Optional[TrainerConfig] = None,
            device: Optional[str] = None,
            scheduler: Optional[Any] = None
    ):
        """Initialize the Trainer.

        Args:
            model: Neural network model to train.
            optimizer: Optimizer for parameter updates.
            criterion: Loss function for training.
            config: Training configuration (uses defaults if None).
            device: Device string like "cuda" or "cpu" (auto-detect if None).
            scheduler: Optional learning rate scheduler.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config or TrainerConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._setup_directories()
        self._initialize_training_state()
        self.model.to(self.device)

        self.current_epoch: int = -1
        self.current_step: int = -1
        self.global_step: int = -1
        self.best_val_loss: float = float("inf")
        self.training_history: list = []

    def _setup_directories(self) -> None:
        """Create directory structure for snapshots and models."""
        base = Path(self.config.base_dir)
        self.model_root_dir = base / self.config.model_name
        self.snapshots_dir = self.model_root_dir / "snapshots"
        self.models_dir = self.model_root_dir / "models"

    def _initialize_training_state(self) -> None:
        """Initialize training state tracking variables."""
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.training_history: List[Dict[str, Any]] = []

    # ================================================================================
    # Training Methods
    # ================================================================================
    def train(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            num_epochs: int = 10,
            start_epoch: int = 0
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (optional).
            num_epochs: Total number of epochs to train.
            start_epoch: Epoch to start from (for resuming training).

        Returns:
            Dictionary with training history and final metrics.
        """
        self.current_epoch = start_epoch
        print(f"Training on {self.device} for {num_epochs} epochs")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            train_loss = self._train_one_epoch(train_loader, epoch)
            val_loss = self._validate(val_loader) if val_loader else None

            self._log_epoch_summary(epoch, train_loss, val_loss)
            self._save_epoch_checkpoint(epoch, train_loss, val_loss)
            self._update_scheduler()

        return self._finalize_training()

    def _train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data.
            epoch: Current epoch number.

        Returns:
            Average training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for step, batch in enumerate(train_loader):
            loss = self._train_one_step(batch)
            total_loss += loss
            self.current_step = step
            self.global_step += 1

            self._log_step_progress(epoch, step, num_batches, loss)
            self._save_step_checkpoint(epoch, step)

        return total_loss / num_batches

    def _train_one_step(self, batch: Dict[str, Tensor]) -> float:
        """Execute one training step (forward, backward, update).

        The loss is computed using Negative Log-Likelihood:
        $$ \\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log P(y_i | x_i) $$

        Args:
            batch: Dictionary containing 'source_ids' and 'target_ids'.

        Returns:
            Loss value for this step.
        """
        source_ids = batch["source_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)

        # Shift target for teacher forcing: input is [:-1], target is [1:]
        decoder_input = target_ids[:, :-1]
        decoder_target = target_ids[:, 1:]

        self.optimizer.zero_grad()
        log_probabilities = self.model.forward(x=source_ids, y=decoder_input)

        loss = self._compute_loss(log_probabilities, decoder_target)
        loss.backward()
        self._clip_gradients()
        self.optimizer.step()

        return loss.item()

    def _compute_loss(self, log_probabilities: Tensor, targets: Tensor) -> Tensor:
        """Compute loss between predictions and targets.

        Args:
            log_probabilities: Log probabilities of shape (B, T, V).
            targets: Target token indices of shape (B, T).

        Returns:
            Scalar loss tensor.
        """
        # Flatten for loss computation: (B, T, V) -> (B*T, V), (B, T) -> (B*T)
        log_probs_flat = log_probabilities.reshape(-1, log_probabilities.size(-1))
        targets_flat = targets.reshape(-1)
        return self.criterion(log_probs_flat, targets_flat)

    def _clip_gradients(self) -> None:
        """Clip gradients to prevent exploding gradients."""
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

    def _update_scheduler(self) -> None:
        """Update learning rate scheduler if present."""
        if self.scheduler is not None:
            self.scheduler.step()

    # ================================================================================
    # Validation Methods
    # ================================================================================
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Evaluate model on validation set.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        for batch in val_loader:
            source_ids = batch["source_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]

            log_probabilities = self.model.forward(x=source_ids, y=decoder_input)
            loss = self._compute_loss(log_probabilities, decoder_target)
            total_loss += loss.item()

        return total_loss / len(val_loader)

    # ================================================================================
    # Logging Methods
    # ================================================================================
    def _log_step_progress(
            self,
            epoch: int,
            step: int,
            num_batches: int,
            loss: float
    ) -> None:
        """Log training progress at step level."""
        if (step + 1) % self.config.log_interval == 0 or step == 0:
            print(f"  Epoch {epoch} | Step {step + 1}/{num_batches} | Loss: {loss:.4f}")

    def _log_epoch_summary(
            self,
            epoch: int,
            train_loss: float,
            val_loss: Optional[float]
    ) -> None:
        """Log summary at end of epoch."""
        msg = f"Epoch {epoch} | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
        print(msg)

        self.training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

    # ================================================================================
    # Checkpoint Methods
    # ================================================================================
    def _save_step_checkpoint(self, epoch: int, step: int) -> None:
        """Save checkpoint at step interval if configured."""
        interval = self.config.snapshot_interval
        if interval > 0 and (step + 1) % interval == 0:
            self.save_snapshot(epoch, step)

    def _save_epoch_checkpoint(
            self,
            epoch: int,
            train_loss: float,
            val_loss: Optional[float]
    ) -> None:
        """Save checkpoint at end of epoch if configured."""
        if not self.config.snapshot_per_epoch:
            return

        self.save_snapshot(epoch, step=0, loss=val_loss or train_loss)

        # Update best model tracking
        current_loss = val_loss if val_loss is not None else train_loss
        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss

        # Cleanup old snapshots
        if self.config.keep_last_n_snapshots > 0:
            self.cleanup_old_snapshots(self.config.keep_last_n_snapshots)

    def save_snapshot(
            self,
            epoch: int,
            step: int = 0,
            loss: Optional[float] = None,
            extra_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save training snapshot for crash recovery.

        Args:
            epoch: Current epoch number.
            step: Current step within epoch.
            loss: Current loss value (optional).
            extra_state: Additional state to save (optional).

        Returns:
            Path to saved snapshot file.
        """
        ensure_directory_exists(self.snapshots_dir)

        checkpoint = self._build_checkpoint_dict(epoch, step, loss, extra_state)
        filename = build_snapshot_filename(epoch, step)
        filepath = self.snapshots_dir / filename

        torch.save(checkpoint, filepath)
        print(f"Snapshot saved: {filepath}")
        return filepath

    def _build_checkpoint_dict(
            self,
            epoch: int,
            step: int,
            loss: Optional[float],
            extra_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build dictionary containing all checkpoint data."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "global_step": self.global_step,
            "loss": loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state
        return checkpoint

    def load_snapshot(
            self,
            filename: str,
            map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load training snapshot to resume training.

        Args:
            filename: Snapshot filename or full path.
            map_location: Device to load tensors to (e.g., "cpu", "cuda:0").

        Returns:
            Checkpoint dictionary with epoch, step, loss, etc.
        """
        filepath = resolve_file_path(filename, self.snapshots_dir)
        checkpoint = torch.load(filepath, map_location=map_location)

        self._restore_from_checkpoint(checkpoint)
        print(f"Snapshot loaded: {filepath} (epoch {self.current_epoch})")
        return checkpoint

    def _restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore all state from checkpoint dictionary."""
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.current_step = checkpoint.get("step", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def find_latest_snapshot(self) -> Optional[Path]:
        """Find the most recent snapshot file.

        Returns:
            Path to latest snapshot, or None if none exist.
        """
        return find_latest_file(self.snapshots_dir, "snapshot_*.pt")

    def cleanup_old_snapshots(self, keep_last_n: int = 5) -> int:
        """Remove old snapshots, keeping only the most recent ones.

        Args:
            keep_last_n: Number of recent snapshots to keep.

        Returns:
            Number of snapshots deleted.
        """
        deleted = cleanup_old_files(self.snapshots_dir, "snapshot_*.pt", keep_last_n)
        if deleted > 0:
            print(f"Cleaned up {deleted} old snapshots")
        return deleted

    # ================================================================================
    # Model Saving Methods
    # ================================================================================
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training: save model and cleanup snapshots."""
        model_path = self.save_model()

        if self.config.delete_snapshots_after_training:
            self._delete_all_snapshots()

        return {
            "model_path": model_path,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "total_epochs": self.current_epoch + 1,
        }

    def save_model(self, filename: Optional[str] = None) -> Path:
        """Save completed model for inference.

        Args:
            filename: Custom filename without .pt extension (optional).

        Returns:
            Path to saved model file.
        """
        ensure_directory_exists(self.models_dir)

        filename = build_model_filename(filename)
        filepath = self.models_dir / filename

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "model_name": self.config.model_name,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "timestamp": datetime.now().isoformat(),
        }

        torch.save(save_dict, filepath)
        print(f"Model saved: {filepath}")
        return filepath

    def load_model(
            self,
            filename: str,
            map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load a saved model for inference or further training.

        Args:
            filename: Model filename or full path.
            map_location: Device to load tensors to.

        Returns:
            Dictionary containing model metadata.
        """
        filepath = resolve_file_path(filename, self.models_dir)
        save_dict = torch.load(filepath, map_location=map_location)

        self.model.load_state_dict(save_dict["model_state_dict"])
        self.best_val_loss = save_dict.get("best_val_loss", float("inf"))
        self.training_history = save_dict.get("training_history", [])

        print(f"Model loaded: {filepath}")
        return save_dict

    def _delete_all_snapshots(self) -> int:
        """Delete all snapshot files after training completion."""
        deleted = delete_files_by_pattern(self.snapshots_dir, "snapshot_*.pt")
        if deleted > 0:
            print(f"Deleted {deleted} training snapshots")
        return deleted


class LanguageModelTrainer(Trainer):
    """Trainer subclass for decoder-only Language Models (GPT-style).

    Unlike Trainer which expects encoder-decoder models with source_ids and
    target_ids, LanguageModelTrainer handles language models where:
    - Batch is a tuple (input_ids, target_ids) where target is input shifted by 1
    - Model.forward(x) takes only input, no source/memory

    Usage:
        from scratch.lm import LanguageModel
        from scratch.trainer import LanguageModelTrainer, TrainerConfig

        model = LanguageModel(vocab_size=50257)
        trainer = LanguageModelTrainer(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            criterion=torch.nn.NLLLoss(),
            config=TrainerConfig(model_name="gpt_wikitext")
        )
        trainer.train(train_loader, val_loader, num_epochs=10)
    """

    def _train_one_step(self, batch) -> float:
        """Execute one training step for language model.

        The loss is computed using Negative Log-Likelihood:
        $$ \\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log P(x_t | x_{<t}) $$

        Args:
            batch: Tuple of (input_ids, target_ids) tensors.

        Returns:
            Loss value for this step.
        """
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        self.optimizer.zero_grad()
        log_probabilities = self.model.forward(input_ids)

        loss = self._compute_loss(log_probabilities, target_ids)
        loss.backward()
        self._clip_gradients()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _validate(self, val_loader) -> float:
        """Evaluate language model on validation set.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        for batch in val_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            log_probabilities = self.model.forward(input_ids)
            loss = self._compute_loss(log_probabilities, target_ids)
            total_loss += loss.item()

        return total_loss / len(val_loader)


if __name__ == "__main__":
    print(__doc__)
