"""Early stopping callback for Trainer.

Stops training when validation loss stops improving, with optional
model weight restoration to the best epoch.

Also supports overfitting detection by monitoring when val_loss - train_loss
keeps growing for N consecutive epochs.

Usage:
    from trainer_early_stopping import EarlyStoppingCallback

    # Standard early stopping (val loss not improving)
    callback = EarlyStoppingCallback(
        patience=5,
        min_delta=0.001,
        restore_best=True
    )

    # With overfitting detection
    callback = EarlyStoppingCallback(
        patience=5,
        min_delta=0.001,
        restore_best=True,
        overfit_patience=3,        # Stop if gap grows for 3 epochs
        overfit_min_delta=0.01     # Minimum gap increase to count
    )

    trainer = Trainer(..., callbacks=[callback])
    trainer.train(...)
"""
from typing import TYPE_CHECKING, Dict, Any, Optional

import torch
from torch import nn

from training.trainer_callback import TrainerCallback

if TYPE_CHECKING:
    from training.trainer import Trainer


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback with two stopping criteria.

    1. Validation loss stops improving (standard early stopping)
    2. Overfitting detection: val_loss - train_loss keeps growing

    Monitors validation loss (or training loss if validation not available) and
    stops training after patience epochs without improvement.

    Also monitors the gap between validation and training loss. If this gap
    keeps increasing for overfit_patience consecutive epochs, it indicates
    overfitting and triggers early stopping.

    Attributes:
        patience: Number of epochs to wait for val loss improvement.
        min_delta: Minimum change to qualify as improvement.
        restore_best: Whether to restore best weights when stopping.
        overfit_patience: Number of epochs to wait before stopping due to overfitting (0 to disable).
        overfit_min_delta: Minimum gap increase to count as overfitting trend.

        counter: Current count of epochs without improvement.
        best_loss: Best loss seen so far.
        best_epoch: Epoch where best loss was achieved.
        best_weights: Stored best model weights (if restore_best=True).

        overfit_counter: Current count of epochs with increasing gap.
        prev_gap: Previous epoch's val_loss - train_loss gap.
        should_stop: Flag indicating if training should stop.
    """

    def __init__(
            self,
            patience: int = 5,
            min_delta: float = 0.001,
            restore_best: bool = True,
            overfit_patience: int = 0,
            overfit_min_delta: float = 0.01
    ):
        """Initialize early stopping with optional overfitting detection.

        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum improvement to reset patience counter.
            restore_best: Whether to save and restore best weights.
            overfit_patience: Number of epochs with increasing val-train gap before stopping (0 to disable).
            overfit_min_delta: Minimum gap increase to count as overfitting trend.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.overfit_patience = overfit_patience
        self.overfit_min_delta = overfit_min_delta

        # Standard early stopping state
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_weights = None

        # Overfitting detection state
        self.overfit_counter = 0
        self.prev_gap = None  # Previous (val_loss - train_loss)

        self._should_stop = False

    def on_epoch_end(
            self,
            trainer: 'Trainer',
            epoch: int,
            train_loss: float,
            val_loss: Optional[float]
    ) -> None:
        """Check if training should stop based on current loss and overfitting.

        Args:
            trainer: The Trainer instance.
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss (None if not available).
        """
        # Use validation loss if available, otherwise training loss
        current_loss = val_loss if val_loss is not None else train_loss

        # === Standard Early Stopping (val loss not improving) ===
        if current_loss < self.best_loss - self.min_delta:
            # Improvement detected
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.counter = 0

            # Save best weights if requested
            if self.restore_best:
                self.best_weights = {
                    k: v.cpu().clone()
                    for k, v in trainer.model.state_dict().items()
                }

            print(f"  Early stopping: new best loss {self.best_loss:.4f} at epoch {epoch}")
        else:
            # No improvement
            self.counter += 1
            print(f"  Early stopping: no improvement for {self.counter}/{self.patience} epochs "
                  f"(best: {self.best_loss:.4f} at epoch {self.best_epoch})")

            if self.counter >= self.patience:
                self._should_stop = True
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"  Reason: No improvement for {self.patience} epochs")
                print(f"  Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")

                if self.restore_best:
                    self._restore_best_weights(trainer.model)
                return

        # === Overfitting Detection (val_loss - train_loss keeps growing) ===
        if self.overfit_patience > 0 and val_loss is not None:
            current_gap = val_loss - train_loss

            if self.prev_gap is not None:
                gap_increase = current_gap - self.prev_gap

                if gap_increase > self.overfit_min_delta:
                    # Gap is increasing (overfitting trend)
                    self.overfit_counter += 1
                    print(f"  Overfitting: gap increasing for {self.overfit_counter}/{self.overfit_patience} epochs "
                          f"(gap: {current_gap:.4f}, increase: +{gap_increase:.4f})")

                    if self.overfit_counter >= self.overfit_patience:
                        self._should_stop = True
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        print(f"  Reason: Overfitting detected (val-train gap growing for {self.overfit_patience} epochs)")
                        print(f"  Current gap: {current_gap:.4f}")
                        print(f"  Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

                        if self.restore_best:
                            self._restore_best_weights(trainer.model)
                        return
                else:
                    # Gap not increasing significantly, reset counter
                    self.overfit_counter = 0

            self.prev_gap = current_gap

    def should_stop_training(self, trainer: 'Trainer') -> bool:
        """Return True if training should stop.

        Args:
            trainer: The Trainer instance.

        Returns:
            True if patience exceeded, False otherwise.
        """
        return self._should_stop

    def _restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best weights to the model.

        Args:
            model: Model to restore weights to.
        """
        if self.best_weights is not None:
            # Move weights to model's device
            device = next(model.parameters()).device
            restored_weights = {
                k: v.to(device)
                for k, v in self.best_weights.items()
            }
            model.load_state_dict(restored_weights)
            print(f"  Restored best weights from epoch {self.best_epoch} (loss: {self.best_loss:.4f})")

    def on_snapshot_save(
            self,
            trainer: 'Trainer',
            epoch: int,
            step: int
    ) -> Dict[str, Any]:
        """Save early stopping state to snapshot.

        Args:
            trainer: The Trainer instance.
            epoch: Current epoch.
            step: Current step.

        Returns:
            Dictionary with early stopping state.
        """
        state = {
            'counter': self.counter,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'should_stop': self._should_stop,
            'overfit_counter': self.overfit_counter,
            'prev_gap': self.prev_gap,
        }

        # Only save best_weights if they exist and restore_best is enabled
        if self.restore_best and self.best_weights is not None:
            state['best_weights'] = self.best_weights

        return state

    def on_snapshot_load(self, trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:
        """Restore early stopping state from snapshot.

        Args:
            trainer: The Trainer instance.
            checkpoint: Checkpoint dictionary containing callback state.
        """
        self.counter = checkpoint.get('counter', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self._should_stop = checkpoint.get('should_stop', False)
        self.overfit_counter = checkpoint.get('overfit_counter', 0)
        self.prev_gap = checkpoint.get('prev_gap', None)

        if 'best_weights' in checkpoint:
            self.best_weights = checkpoint['best_weights']

        print(f"  Early stopping state restored: "
              f"best_loss={self.best_loss:.4f}, "
              f"counter={self.counter}/{self.patience}, "
              f"overfit_counter={self.overfit_counter}/{self.overfit_patience}")
