"""Gradient flow monitoring callback for Trainer.

Monitors gradient flow through model layers to detect vanishing/exploding
gradients. Synchronized with snapshot saving to ensure saved snapshots
correspond to monitored states.

Usage:
    from trainer_gradient_monitor import GradientMonitorCallback

    callback = GradientMonitorCallback(
        monitor_at_snapshots=True,  # Monitor when snapshots are saved
        monitor_interval=100         # Or monitor every N steps
    )

    trainer = Trainer(..., callbacks=[callback])
    trainer.train(...)
"""
from typing import TYPE_CHECKING, Dict, Any, Optional

from torch import nn

from trainer_callback import TrainerCallback
from gradient_monitor import GradientGainMonitor

if TYPE_CHECKING:
    from trainer import Trainer


class GradientMonitorCallback(TrainerCallback):
    """Gradient flow monitoring callback.

    Monitors gradient flow through model layers using GradientGainMonitor.
    Can be configured to monitor:
    - At snapshot save intervals (synced with checkpointing)
    - At fixed step intervals
    - At epoch boundaries

    Detects monitorable blocks in the model:
    - model.decoder.layers (decoder-only LM)
    - model.encoder.layers (encoder)
    - model.blocks (custom models)
    - model.layers (direct layers)

    Attributes:
        monitor_at_snapshots: Monitor when snapshots are saved.
        monitor_interval: Monitor every N steps (0 to disable).
        monitor_at_epochs: Monitor at end of each epoch.
        norm_type: Norm type for gradient measurement ('l2', 'l1', 'linf', 'mean').
    """

    def __init__(
            self,
            monitor_at_snapshots: bool = True,
            monitor_interval: int = 0,
            monitor_at_epochs: bool = False,
            norm_type: str = 'l2'
    ):
        """Initialize gradient monitor callback.

        Args:
            monitor_at_snapshots: Monitor when snapshots are saved (synced).
            monitor_interval: Monitor every N steps (0 to disable).
            monitor_at_epochs: Monitor at end of each epoch.
            norm_type: Norm type for gradient measurement.
        """
        self.monitor_at_snapshots = monitor_at_snapshots
        self.monitor_interval = monitor_interval
        self.monitor_at_epochs = monitor_at_epochs
        self.norm_type = norm_type

        self.gradient_monitor = None
        self.monitorable_blocks = None
        self.block_name = None
        self.current_stats = None  # Stats from most recent monitoring

    def on_train_start(self, trainer: 'Trainer') -> None:
        """Initialize gradient monitor with model structure.

        Args:
            trainer: The Trainer instance.
        """
        # Detect monitorable blocks
        self.monitorable_blocks = self._detect_monitorable_blocks(trainer.model)

        if not self.monitorable_blocks:
            print("Warning: Gradient monitor enabled but no monitorable blocks found")
            print("  Expected: model.decoder.layers, model.encoder.layers, model.blocks, or model.layers")
            return

        # Use the first available block group
        # TODO: Support monitoring multiple block groups
        self.block_name, blocks = next(iter(self.monitorable_blocks.items()))

        # Initialize monitor
        self.gradient_monitor = GradientGainMonitor(
            blocks,
            norm_type=self.norm_type,
            strict_single_backward=False
        )

        print(f"Gradient flow monitoring enabled: {self.block_name} ({len(blocks)} layers)")
        print(f"  Monitor at snapshots: {self.monitor_at_snapshots}")
        print(f"  Monitor interval: {self.monitor_interval if self.monitor_interval > 0 else 'disabled'}")
        print(f"  Monitor at epochs: {self.monitor_at_epochs}")
        print(f"  Norm type: {self.norm_type}")

    def on_backward_end(self, trainer: 'Trainer') -> None:
        """Capture gradient statistics after backward pass.

        This is called after every backward pass, but we only record stats
        if we're monitoring at this step.

        Args:
            trainer: The Trainer instance.
        """
        if self.gradient_monitor is None:
            return

        # Check if we should monitor at this step
        should_monitor = False

        # Monitor at snapshot intervals (synced with checkpointing)
        if self.monitor_at_snapshots:
            interval = trainer.config.snapshot_interval
            if interval > 0 and (trainer.current_step + 1) % interval == 0:
                should_monitor = True

        # Monitor at fixed intervals
        if self.monitor_interval > 0 and trainer.global_step % self.monitor_interval == 0:
            should_monitor = True

        if not should_monitor:
            # Always reset monitor even if not capturing stats
            self.gradient_monitor.reset()
            return

        # Capture gradient statistics
        try:
            self.current_stats = self.gradient_monitor.summary_stats()

            # Log to TensorBoard
            self._log_to_tensorboard(trainer)

            # Print warnings if issues detected
            self._check_gradient_issues(trainer)

        finally:
            # Always reset for next backward pass
            self.gradient_monitor.reset()

    def on_epoch_end(
            self,
            trainer: 'Trainer',
            epoch: int,
            train_loss: float,
            val_loss: Optional[float]
    ) -> None:
        """Monitor gradient flow at epoch end if configured.

        Args:
            trainer: The Trainer instance.
            epoch: Current epoch number.
            train_loss: Training loss.
            val_loss: Validation loss (if available).
        """
        if not self.monitor_at_epochs or self.gradient_monitor is None:
            return

        # Stats are already captured from last batch's backward_end
        if self.current_stats:
            print(f"\n  Gradient Flow Summary (Epoch {epoch}):")
            print("  " + "-" * 68)
            self._print_summary(self.current_stats)

    def on_snapshot_save(
            self,
            trainer: 'Trainer',
            epoch: int,
            step: int
    ) -> Dict[str, Any]:
        """Save gradient monitoring stats with snapshot.

        This ensures the snapshot contains gradient analysis for the
        exact model state being saved.

        Args:
            trainer: The Trainer instance.
            epoch: Current epoch.
            step: Current step.

        Returns:
            Dictionary with gradient statistics.
        """
        if self.current_stats is None:
            return None

        state = {
            'block_name': self.block_name,
            'stats': self.current_stats,
            'epoch': epoch,
            'step': step,
            'global_step': trainer.global_step,
        }

        # Print snapshot gradient summary
        print(f"  Gradient flow stats saved with snapshot:")
        print(f"    Block: {self.block_name}")
        print(f"    Mean gain: {self.current_stats['mean_gain']:.4f}")
        print(f"    Healthy: {self.current_stats['num_healthy']}, "
              f"Damping: {self.current_stats['num_damping']}, "
              f"Amplifying: {self.current_stats['num_amplifying']}")

        return state

    def on_train_end(self, trainer: 'Trainer', result: Dict[str, Any]) -> None:
        """Clean up gradient monitor.

        Args:
            trainer: The Trainer instance.
            result: Training result dictionary.
        """
        if self.gradient_monitor is not None:
            self.gradient_monitor.close()
            print("Gradient monitor closed")

    def _detect_monitorable_blocks(self, model: nn.Module) -> Dict[str, nn.ModuleList]:
        """Detect monitorable layer blocks in the model.

        Args:
            model: The model to inspect.

        Returns:
            Dictionary mapping block name to ModuleList.
        """
        monitorable = {}

        # Check for decoder.layers (LanguageModel, Decoder-only Transformer)
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            if isinstance(model.decoder.layers, nn.ModuleList):
                monitorable['decoder'] = model.decoder.layers

        # Check for encoder.layers (Encoder-Decoder Transformer)
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            if isinstance(model.encoder.layers, nn.ModuleList):
                monitorable['encoder'] = model.encoder.layers

        # Check for direct blocks attribute
        if hasattr(model, 'blocks') and isinstance(model.blocks, nn.ModuleList):
            monitorable['blocks'] = model.blocks

        # Check for direct layers attribute
        if hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
            monitorable['layers'] = model.layers

        return monitorable

    def _log_to_tensorboard(self, trainer: 'Trainer') -> None:
        """Log gradient statistics to TensorBoard.

        Args:
            trainer: The Trainer instance.
        """
        if self.current_stats is None:
            return

        step = trainer.global_step
        stats = self.current_stats

        trainer.writer.add_scalar(f"gradient_monitor/mean_gain", stats['mean_gain'], step)
        trainer.writer.add_scalar(f"gradient_monitor/min_gain", stats['min_gain'], step)
        trainer.writer.add_scalar(f"gradient_monitor/max_gain", stats['max_gain'], step)
        trainer.writer.add_scalar(f"gradient_monitor/mean_log_gain", stats['mean_log_gain'], step)
        trainer.writer.add_scalar(f"gradient_monitor/num_healthy", stats['num_healthy'], step)
        trainer.writer.add_scalar(f"gradient_monitor/num_damping", stats['num_damping'], step)
        trainer.writer.add_scalar(f"gradient_monitor/num_amplifying", stats['num_amplifying'], step)
        trainer.writer.add_scalar(f"gradient_monitor/num_vanishing", stats['num_vanishing'], step)

    def _check_gradient_issues(self, trainer: 'Trainer') -> None:
        """Check for gradient issues and print warnings.

        Args:
            trainer: The Trainer instance.
        """
        if self.current_stats is None or self.gradient_monitor is None:
            return

        stats = self.current_stats
        num_layers = len(next(iter(self.monitorable_blocks.values())))

        # Warning for vanishing gradients
        if stats['num_vanishing'] > 0:
            print(f"  ⚠️  WARNING (step {trainer.global_step}): "
                  f"{stats['num_vanishing']} vanishing gradient transitions!")

        # Warning for many damping transitions
        if stats['num_damping'] > num_layers // 2:
            print(f"  ⚠️  WARNING (step {trainer.global_step}): "
                  f"Many damping transitions ({stats['num_damping']}/{num_layers-1}) "
                  f"- possible vanishing gradients")

        # Warning for many amplifying transitions
        if stats['num_amplifying'] > num_layers // 3:
            print(f"  ⚠️  WARNING (step {trainer.global_step}): "
                  f"Many amplifying transitions ({stats['num_amplifying']}/{num_layers-1}) "
                  f"- check for exploding gradients")

    def _print_summary(self, stats: Dict[str, Any]) -> None:
        """Print gradient flow summary.

        Args:
            stats: Gradient statistics dictionary.
        """
        print(f"  Mean gain: {stats['mean_gain']:.4f}")
        print(f"  Min gain: {stats['min_gain']:.4f}")
        print(f"  Max gain: {stats['max_gain']:.4f}")
        print(f"  Healthy: {stats['num_healthy']}, "
              f"Damping: {stats['num_damping']}, "
              f"Amplifying: {stats['num_amplifying']}, "
              f"Vanishing: {stats['num_vanishing']}")
