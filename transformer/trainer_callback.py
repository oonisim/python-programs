"""Base callback interface for Trainer plugins.

This module defines the callback interface that allows extending Trainer
functionality without modifying the core training loop. Similar to TensorFlow/Keras
callbacks or PyTorch Lightning hooks.

Usage:
    class MyCallback(TrainerCallback):
        def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
            print(f"Epoch {epoch} completed")

    trainer = Trainer(..., callbacks=[MyCallback()])
"""
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from trainer import Trainer


class TrainerCallback:
    """Base class for trainer callbacks/plugins.

    Callbacks allow you to inject custom behavior at specific points in the
    training loop without modifying the Trainer class itself.

    All methods receive the trainer instance, allowing access to:
    - trainer.model
    - trainer.optimizer
    - trainer.config
    - trainer.writer (TensorBoard)
    - trainer.current_epoch, trainer.global_step
    - etc.

    Hook Points (in execution order):
    1. on_train_start() - Before training begins
    2. on_epoch_start() - Before each epoch
    3. on_batch_start() - Before each batch
    4. on_forward_end() - After forward pass, before backward
    5. on_backward_end() - After backward pass, before gradient clipping
    6. on_step_end() - After optimizer.step()
    7. on_batch_end() - After batch processing
    8. on_epoch_end() - After each epoch (after validation)
    9. on_train_end() - After training completes

    Special hooks:
    - on_snapshot_save() - When saving snapshot
    - should_stop_training() - Check if training should stop early
    """

    def on_train_start(self, trainer: 'Trainer') -> None:
        """Called once before training begins.

        Args:
            trainer: The Trainer instance.
        """
        pass

    def on_train_end(self, trainer: 'Trainer', result: Dict[str, Any]) -> None:
        """Called once after training completes.

        Args:
            trainer: The Trainer instance.
            result: Training result dictionary.
        """
        pass

    def on_epoch_start(self, trainer: 'Trainer', epoch: int) -> None:
        """Called at the start of each epoch.

        Args:
            trainer: The Trainer instance.
            epoch: Current epoch number.
        """
        pass

    def on_epoch_end(
            self,
            trainer: 'Trainer',
            epoch: int,
            train_loss: float,
            val_loss: Optional[float]
    ) -> None:
        """Called at the end of each epoch, after validation.

        Args:
            trainer: The Trainer instance.
            epoch: Current epoch number.
            train_loss: Average training loss for this epoch.
            val_loss: Validation loss (None if no validation).
        """
        pass

    def on_batch_start(self, trainer: 'Trainer', batch_idx: int) -> None:
        """Called before processing each batch.

        Args:
            trainer: The Trainer instance.
            batch_idx: Current batch index.
        """
        pass

    def on_batch_end(self, trainer: 'Trainer', batch_idx: int, loss: float) -> None:
        """Called after processing each batch.

        Args:
            trainer: The Trainer instance.
            batch_idx: Current batch index.
            loss: Loss value for this batch.
        """
        pass

    def on_forward_end(self, trainer: 'Trainer', loss: Any) -> None:
        """Called after forward pass, before backward.

        Args:
            trainer: The Trainer instance.
            loss: Loss tensor (before .backward()).
        """
        pass

    def on_backward_end(self, trainer: 'Trainer') -> None:
        """Called after backward pass, before gradient clipping.

        This is the ideal point for gradient analysis (gradients are computed
        but not yet clipped or cleared).

        Args:
            trainer: The Trainer instance.
        """
        pass

    def on_step_end(self, trainer: 'Trainer') -> None:
        """Called after optimizer.step(), before gradients are zeroed.

        Args:
            trainer: The Trainer instance.
        """
        pass

    def on_snapshot_save(
            self,
            trainer: 'Trainer',
            epoch: int,
            step: int
    ) -> Optional[Dict[str, Any]]:
        """Called when saving a snapshot.

        Returns extra state to include in the snapshot.

        Args:
            trainer: The Trainer instance.
            epoch: Current epoch.
            step: Current step.

        Returns:
            Optional dictionary of extra state to save in snapshot.
        """
        return None

    def should_stop_training(self, trainer: 'Trainer') -> bool:
        """Check if training should stop early.

        Called at the end of each epoch. If any callback returns True,
        training stops.

        Args:
            trainer: The Trainer instance.

        Returns:
            True if training should stop, False otherwise.
        """
        return False

    def on_snapshot_load(self, trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:
        """Called after loading a snapshot.

        Allows callback to restore its state from checkpoint.

        Args:
            trainer: The Trainer instance.
            checkpoint: Loaded checkpoint dictionary.
        """
        pass


class CallbackList:
    """Container for managing multiple callbacks."""

    def __init__(self, callbacks: list):
        """Initialize callback list.

        Args:
            callbacks: List of TrainerCallback instances.
        """
        self.callbacks = callbacks or []

    def on_train_start(self, trainer: 'Trainer') -> None:
        """Call on_train_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_start(trainer)

    def on_train_end(self, trainer: 'Trainer', result: Dict[str, Any]) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer, result)

    def on_epoch_start(self, trainer: 'Trainer', epoch: int) -> None:
        """Call on_epoch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_start(trainer, epoch)

    def on_epoch_end(
            self,
            trainer: 'Trainer',
            epoch: int,
            train_loss: float,
            val_loss: Optional[float]
    ) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, train_loss, val_loss)

    def on_batch_start(self, trainer: 'Trainer', batch_idx: int) -> None:
        """Call on_batch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_start(trainer, batch_idx)

    def on_batch_end(self, trainer: 'Trainer', batch_idx: int, loss: float) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, loss)

    def on_forward_end(self, trainer: 'Trainer', loss: Any) -> None:
        """Call on_forward_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_forward_end(trainer, loss)

    def on_backward_end(self, trainer: 'Trainer') -> None:
        """Call on_backward_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_backward_end(trainer)

    def on_step_end(self, trainer: 'Trainer') -> None:
        """Call on_step_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_step_end(trainer)

    def on_snapshot_save(
            self,
            trainer: 'Trainer',
            epoch: int,
            step: int
    ) -> Dict[str, Any]:
        """Collect extra state from all callbacks for snapshot.

        Returns:
            Dictionary mapping callback class name to its state.
        """
        extra_state = {}
        for callback in self.callbacks:
            state = callback.on_snapshot_save(trainer, epoch, step)
            if state is not None:
                callback_name = callback.__class__.__name__
                extra_state[callback_name] = state
        return extra_state

    def should_stop_training(self, trainer: 'Trainer') -> bool:
        """Check if any callback wants to stop training.

        Returns:
            True if any callback returns True, False otherwise.
        """
        return any(callback.should_stop_training(trainer) for callback in self.callbacks)

    def on_snapshot_load(self, trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:
        """Allow callbacks to restore state from checkpoint."""
        extra_state = checkpoint.get('callback_state', {})
        for callback in self.callbacks:
            callback_name = callback.__class__.__name__
            if callback_name in extra_state:
                callback.on_snapshot_load(trainer, extra_state[callback_name])
