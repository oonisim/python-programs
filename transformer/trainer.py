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
from torch.utils.tensorboard import SummaryWriter

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
        early_stop_patience: Stop training after N epochs without improvement (0 to disable).
        early_stop_min_delta: Minimum change to qualify as improvement.
        early_stop_restore_best: Restore best model weights when early stopping triggers.
    """
    model_name: str = "transformer"
    base_dir: str = "."
    gradient_clip: Optional[float] = 1.0
    log_interval: int = 100
    snapshot_interval: int = 0
    snapshot_per_epoch: bool = True
    keep_last_n_snapshots: int = 5
    delete_snapshots_after_training: bool = True
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.001
    early_stop_restore_best: bool = True


class EarlyStopping:
    """Early stopping handler to stop training when validation loss stops improving.

    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        restore_best: Whether to restore best weights when stopping.
        counter: Current count of epochs without improvement.
        best_loss: Best loss seen so far.
        best_epoch: Epoch where best loss was achieved.
        best_weights: Stored best model weights (if restore_best=True).
        should_stop: Flag indicating if training should stop.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001, restore_best: bool = True):
        """Initialize early stopping.

        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum improvement to reset patience counter.
            restore_best: Whether to save and restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_weights = None
        self.should_stop = False

    def __call__(self, current_loss: float, epoch: int, model: nn.Module) -> bool:
        """Check if training should stop based on current loss.

        Args:
            current_loss: Current validation/training loss.
            epoch: Current epoch number.
            model: Model to potentially save weights from.

        Returns:
            True if training should stop, False otherwise.
        """
        # Check if this is an improvement
        if current_loss < self.best_loss - self.min_delta:
            # Improvement detected
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.counter = 0

            # Save best weights if requested
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            return False
        else:
            # No improvement
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True
                return True

            return False

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best weights to the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"  Restored best weights from epoch {self.best_epoch} (loss: {self.best_loss:.4f})")


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

        self.writer = SummaryWriter(log_dir=self.model_root_dir / "runs")

        # Initialize early stopping if enabled
        self.early_stopping = None
        if self.config.early_stop_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stop_patience,
                min_delta=self.config.early_stop_min_delta,
                restore_best=self.config.early_stop_restore_best
            )

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
        early_stop_msg = ""
        if self.early_stopping:
            early_stop_msg = f" (early stopping: patience={self.config.early_stop_patience}, min_delta={self.config.early_stop_min_delta})"
        print(f"Training on {self.device} for {num_epochs} epochs{early_stop_msg}")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            train_loss = self._train_one_epoch(train_loader, epoch)
            val_loss = self._validate(val_loader) if val_loader else None

            self._log_epoch_summary(epoch, train_loss, val_loss)
            self._save_epoch_checkpoint(epoch, train_loss, val_loss)
            self._update_scheduler()

            # Check early stopping
            if self.early_stopping:
                # Use validation loss if available, otherwise training loss
                monitor_loss = val_loss if val_loss is not None else train_loss
                should_stop = self.early_stopping(monitor_loss, epoch, self.model)

                if should_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"  No improvement for {self.config.early_stop_patience} epochs")
                    print(f"  Best loss: {self.early_stopping.best_loss:.4f} at epoch {self.early_stopping.best_epoch}")

                    if self.config.early_stop_restore_best:
                        self.early_stopping.restore_best_weights(self.model)

                    break

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
    def _check_weights_valid(self) -> tuple[bool, list[str]]:
        """Check if any model weights contain NaN or Inf values.

        Returns:
            Tuple of (is_valid, list_of_invalid_params)
        """
        invalid_params = []

        for name, param in self.model.named_parameters():
            if torch.isnan(param.data).any():
                invalid_params.append(f"{name} contains NaN")
            if torch.isinf(param.data).any():
                invalid_params.append(f"{name} contains Inf")

        return len(invalid_params) == 0, invalid_params

    def _check_gradients_valid(self) -> tuple[bool, list[str]]:
        """Check if any gradients contain NaN or Inf values.

        Returns:
            Tuple of (is_valid, list_of_invalid_grads)
        """
        invalid_grads = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    invalid_grads.append(f"{name}.grad contains NaN")
                if torch.isinf(param.grad).any():
                    invalid_grads.append(f"{name}.grad contains Inf")

        return len(invalid_grads) == 0, invalid_grads

    def _log_step_progress(
            self,
            epoch: int,
            step: int,
            num_batches: int,
            loss: float
    ) -> None:
        """Log training progress at step level."""
        self.writer.add_scalar("train/step_loss", loss, self.global_step)

        # Periodic weight and gradient validation every 1000 steps
        if self.global_step % 1000 == 0 and self.global_step > 0:
            is_valid_weights, invalid_params = self._check_weights_valid()
            is_valid_grads, invalid_grads = self._check_gradients_valid()

            if not is_valid_weights:
                print(f"  WARNING at step {self.global_step}: Invalid weights detected")
                for msg in invalid_params:
                    print(f"    - {msg}")

            if not is_valid_grads:
                print(f"  WARNING at step {self.global_step}: Invalid gradients detected")
                for msg in invalid_grads:
                    print(f"    - {msg}")

            if is_valid_weights and is_valid_grads:
                print(f"  ✓ Validation at step {self.global_step}: All weights and gradients valid")

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
        self.writer.add_scalar("train/epoch_loss", train_loss, epoch)
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
            self.writer.add_scalar("val/epoch_loss", val_loss, epoch)
        print(msg)

        self.training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        self._log_weight_stats(epoch)
        self._log_gradient_flow(epoch)
        self._log_weight_updates(epoch)

    def _log_weight_stats(self, epoch: int) -> None:
        """Log per-layer weight histograms, variance, and validity to TensorBoard."""
        nan_param_count = 0
        inf_param_count = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"weights/{name}", param.data, epoch)
                self.writer.add_scalar(
                    f"weights_variance/{name}", param.data.var().item(), epoch
                )

                # Check for NaN/Inf in weights
                if torch.isnan(param.data).any():
                    nan_param_count += 1
                if torch.isinf(param.data).any():
                    inf_param_count += 1

                if param.grad is not None:
                    self.writer.add_histogram(
                        f"gradients/{name}", param.grad.data, epoch
                    )

        # Log aggregate validity statistics
        self.writer.add_scalar("debug/nan_param_count", nan_param_count, epoch)
        self.writer.add_scalar("debug/inf_param_count", inf_param_count, epoch)

    def _log_gradient_flow(self, epoch: int) -> None:
        """Monitor gradient flow through the network to detect vanishing/exploding gradients."""
        grad_stats = {
            'min_grad': float('inf'),
            'max_grad': float('-inf'),
            'mean_grad': 0.0,
            'total_params': 0
        }

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_abs = param.grad.abs()
                grad_mean = grad_abs.mean().item()
                grad_max = grad_abs.max().item()
                grad_min = grad_abs.min().item()

                # Track extremes
                grad_stats['min_grad'] = min(grad_stats['min_grad'], grad_min)
                grad_stats['max_grad'] = max(grad_stats['max_grad'], grad_max)
                grad_stats['mean_grad'] += grad_mean
                grad_stats['total_params'] += 1

                # Detect dead neurons (zero gradients)
                zero_ratio = (grad_abs < 1e-8).float().mean().item()
                if zero_ratio > 0.9:  # More than 90% zeros
                    print(f"  WARNING: {name} has {zero_ratio*100:.1f}% zero gradients")

                # Log per-layer gradient norms
                grad_norm = torch.norm(param.grad).item()
                self.writer.add_scalar(f"gradient_norms/{name}", grad_norm, epoch)

        # Log aggregate statistics
        if grad_stats['total_params'] > 0:
            grad_stats['mean_grad'] /= grad_stats['total_params']

            self.writer.add_scalar("debug/min_gradient", grad_stats['min_grad'], epoch)
            self.writer.add_scalar("debug/max_gradient", grad_stats['max_grad'], epoch)
            self.writer.add_scalar("debug/mean_gradient", grad_stats['mean_grad'], epoch)

            # Warning for vanishing gradients
            if grad_stats['mean_grad'] < 1e-7:
                print(f"  WARNING: Mean gradient is very small ({grad_stats['mean_grad']:.2e}) - possible vanishing gradients")

            # Warning for exploding gradients
            if grad_stats['max_grad'] > 100:
                print(f"  WARNING: Max gradient is very large ({grad_stats['max_grad']:.2e}) - possible exploding gradients")

    def _log_weight_updates(self, epoch: int) -> None:
        """Monitor how much weights are changing to detect frozen layers."""
        if not hasattr(self, '_prev_weights'):
            # Store initial weights
            self._prev_weights = {name: param.data.clone()
                                 for name, param in self.model.named_parameters()}
            return

        total_change = 0.0
        no_change_count = 0
        param_count = 0

        for name, param in self.model.named_parameters():
            if name in self._prev_weights:
                # Calculate relative change
                diff = (param.data - self._prev_weights[name]).abs()
                relative_change = (diff / (self._prev_weights[name].abs() + 1e-8)).mean().item()

                # Log per-layer changes
                self.writer.add_scalar(f"weight_updates/{name}", relative_change, epoch)
                total_change += relative_change
                param_count += 1

                # Detect frozen layers
                if relative_change < 1e-6:
                    no_change_count += 1
                    print(f"  WARNING: {name} weights barely changed ({relative_change:.2e})")

                # Update stored weights
                self._prev_weights[name] = param.data.clone()

        if param_count > 0:
            avg_change = total_change / param_count
            self.writer.add_scalar("debug/avg_weight_change", avg_change, epoch)

            if no_change_count > param_count * 0.5:
                print(f"  WARNING: {no_change_count}/{param_count} layers have minimal weight updates - check learning rate")

    # ================================================================================
    # Checkpoint Methods
    # ================================================================================
    def _save_step_checkpoint(self, epoch: int, step: int) -> None:
        """Save checkpoint at step interval if configured."""
        interval = self.config.snapshot_interval
        if interval > 0 and (step + 1) % interval == 0:
            self.save_snapshot(epoch, step)
            if self.config.keep_last_n_snapshots > 0:
                self.cleanup_old_snapshots(self.config.keep_last_n_snapshots)

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

        self.writer.close()

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
