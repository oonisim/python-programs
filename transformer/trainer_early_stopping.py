"""Early stopping callback for Trainer.

Stops training when validation loss stops improving, with optional
model weight restoration to the best epoch.

Usage:
    from trainer_early_stopping import EarlyStoppingCallback

    callback = EarlyStoppingCallback(
        patience=5,
        min_delta=0.001,
        restore_best=True
    )

    trainer = Trainer(..., callbacks=[callback])
    trainer.train(...)
"""
from typing import TYPE_CHECKING, Dict, Any, Optional

import torch
from torch import nn

from trainer_callback import TrainerCallback

if TYPE_CHECKING:
    from trainer import Trainer


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback to stop training when validation loss stops improving.

    Monitors validation loss (or training loss if validation not available) and
    stops training after patience epochs without improvement.

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

    def __init__(
            self,
            patience: int = 5,
            min_delta: float = 0.001,
            restore_best: bool = True
    ):
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
        self._should_stop = False

    def on_epoch_end(
            self,
            trainer: 'Trainer',
            epoch: int,
            train_loss: float,
            val_loss: Optional[float]
    ) -> None:
        """Check if training should stop based on current loss.

        Args:
            trainer: The Trainer instance.
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss (None if not available).
        """
        # Use validation loss if available, otherwise training loss
        current_loss = val_loss if val_loss is not None else train_loss

        # Check if this is an improvement
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
                print(f"  No improvement for {self.patience} epochs")
                print(f"  Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")

                if self.restore_best:
                    self._restore_best_weights(trainer.model)

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

        if 'best_weights' in checkpoint:
            self.best_weights = checkpoint['best_weights']

        print(f"  Early stopping state restored: "
              f"best_loss={self.best_loss:.4f}, "
              f"counter={self.counter}/{self.patience}")
