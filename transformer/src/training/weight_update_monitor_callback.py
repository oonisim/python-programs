"""Weight update monitoring callback for Trainer.

Monitors gradient health and actual parameter updates to detect training issues
like vanishing/exploding gradients and frozen parameters. Uses sampling to avoid
memory overhead from full parameter clones.

Usage:
    from training.weight_update_monitor_callback import WeightUpdateMonitorCallback

    callback = WeightUpdateMonitorCallback(
        monitor_interval=100,           # Monitor every N steps
        sample_size=1024,                # Elements sampled per param
        vanishing_grad_threshold=1e-7,
        exploding_grad_threshold=1e2,
    )

    trainer = Trainer(..., callbacks=[callback])
    trainer.train(...)
"""
from typing import TYPE_CHECKING, Dict, Any, Optional

from training.trainer_callback import TrainerCallback
from training.weight_update_monitor import (
    WeightUpdateMonitor,
    GradientDiagnostics,
    UpdateDiagnostics,
)

if TYPE_CHECKING:
    from training.trainer import Trainer


class WeightUpdateMonitorCallback(TrainerCallback):
    """Weight update monitoring callback.

    Monitors gradient statistics and parameter updates using WeightUpdateMonitor.
    Can be configured to monitor at fixed step intervals.

    Attributes:
        monitor_interval: Monitor every N steps (must be >= 1).
        sample_size: Number of elements sampled per parameter.
        vanishing_grad_threshold: Threshold for detecting vanishing gradients.
        exploding_grad_threshold: Threshold for detecting exploding gradients.
        frozen_update_ratio_threshold: Threshold for detecting frozen parameters.
        frozen_patience_steps: Number of consecutive frozen steps before flagging.
        monitor_topk: Number of worst parameters to log to TensorBoard.
    """

    def __init__(
        self,
        monitor_interval: int = 100,
        sample_size: int = 1024,
        vanishing_grad_threshold: float = 1e-7,
        exploding_grad_threshold: float = 1e2,
        frozen_update_ratio_threshold: float = 1e-12,
        frozen_patience_steps: int = 3,
        monitor_topk: int = 5,
    ):
        """Initialize weight update monitor callback.

        Args:
            monitor_interval: Monitor every N steps (must be >= 1).
            sample_size: Elements sampled per parameter (~4KB each).
            vanishing_grad_threshold: Flag if ||grad|| <= this.
            exploding_grad_threshold: Flag if ||grad|| >= this.
            frozen_update_ratio_threshold: Flag if update_ratio <= this.
            frozen_patience_steps: Consecutive frozen steps before flagging.
            monitor_topk: Log top-K worst params (avoids tag explosion).
        """
        if monitor_interval < 1:
            raise ValueError(f"monitor_interval must be >= 1, got {monitor_interval}")

        self.monitor_interval = monitor_interval
        self.sample_size = sample_size
        self.vanishing_grad_threshold = vanishing_grad_threshold
        self.exploding_grad_threshold = exploding_grad_threshold
        self.frozen_update_ratio_threshold = frozen_update_ratio_threshold
        self.frozen_patience_steps = frozen_patience_steps
        self.monitor_topk = monitor_topk

        self.monitor: Optional[WeightUpdateMonitor] = None
        self.current_grad_diag: Optional[Dict[str, GradientDiagnostics]] = None
        self.current_update_diag: Optional[Dict[str, UpdateDiagnostics]] = None

    def on_train_start(self, trainer: 'Trainer') -> None:
        """Initialize weight update monitor.

        Args:
            trainer: The Trainer instance.
        """
        self.monitor = WeightUpdateMonitor(
            sample_size=self.sample_size,
            vanishing_grad_threshold=self.vanishing_grad_threshold,
            exploding_grad_threshold=self.exploding_grad_threshold,
            frozen_update_ratio_threshold=self.frozen_update_ratio_threshold,
            frozen_patience_steps=self.frozen_patience_steps,
        )

        print(
            f"Weight update monitoring enabled "
            f"(sample_size={self.sample_size}, "
            f"interval={self.monitor_interval} steps)"
        )

    def on_backward_end(self, trainer: 'Trainer') -> None:
        """Capture gradient statistics after backward pass.

        Called after backward but before gradient clipping. We capture
        gradients only at monitoring intervals.

        Args:
            trainer: The Trainer instance.
        """
        if self.monitor is None:
            return

        # Only monitor at specified intervals
        if trainer.global_step % self.monitor_interval != 0:
            return

        # Capture gradient diagnostics
        self.current_grad_diag = self.monitor.check_gradients(
            model=trainer.model,
            optimizer=trainer.optimizer,
        )

        # Log gradient statistics
        self._log_gradient_stats(trainer)

    def on_step_end(self, trainer: 'Trainer') -> None:
        """Capture weight update statistics after optimizer step.

        Called after optimizer.step(). We capture updates only at
        monitoring intervals.

        Args:
            trainer: The Trainer instance.
        """
        if self.monitor is None:
            return

        # Only monitor at specified intervals
        if trainer.global_step % self.monitor_interval != 0:
            return

        # Capture update diagnostics
        self.current_update_diag = self.monitor.check_updates(
            model=trainer.model,
            optimizer=trainer.optimizer,
        )

        # Log update statistics
        self._log_update_stats(trainer)

    def on_snapshot_save(
        self,
        trainer: 'Trainer',
        epoch: int,
        step: int,
    ) -> Optional[Dict[str, Any]]:
        """Save weight monitor state with snapshot.

        Args:
            trainer: The Trainer instance.
            epoch: Current epoch.
            step: Current step.

        Returns:
            Dictionary with monitor state.
        """
        if self.monitor is None:
            return None

        # Save internal state for resuming
        state = {
            'frozen_steps': dict(self.monitor._frozen_steps),
            'epoch': epoch,
            'step': step,
            'global_step': trainer.global_step,
        }

        return state

    def on_snapshot_load(self, trainer: 'Trainer', checkpoint: Dict[str, Any]) -> None:
        """Restore weight monitor state from snapshot.

        Args:
            trainer: The Trainer instance.
            checkpoint: Loaded checkpoint dictionary.
        """
        if self.monitor is None:
            return

        # Restore frozen steps tracking
        if 'frozen_steps' in checkpoint:
            self.monitor._frozen_steps = dict(checkpoint['frozen_steps'])
            print(f"  Weight monitor state restored (tracking {len(self.monitor._frozen_steps)} parameters)")

    def on_train_end(self, trainer: 'Trainer', result: Dict[str, Any]) -> None:
        """Clean up weight monitor.

        Args:
            trainer: The Trainer instance.
            result: Training result dictionary.
        """
        if self.monitor is not None:
            # Reset internal state
            self.monitor.reset()
            print("Weight monitor closed")

    def _log_gradient_stats(self, trainer: 'Trainer') -> None:
        """Log gradient statistics to TensorBoard.

        Args:
            trainer: The Trainer instance.
        """
        if self.current_grad_diag is None:
            return

        step = trainer.global_step
        stats = WeightUpdateMonitor.aggregate_gradient_stats(self.current_grad_diag)

        if stats.get("count", 0.0) == 0.0:
            return

        # Log aggregate statistics
        trainer.writer.add_scalar("monitor/grad_norm_median", stats["median"], step)
        trainer.writer.add_scalar("monitor/grad_norm_p95", stats["p95"], step)
        trainer.writer.add_scalar("monitor/grad_norm_min", stats["min"], step)
        trainer.writer.add_scalar("monitor/grad_norm_max", stats["max"], step)
        trainer.writer.add_scalar("monitor/vanishing_count", stats["vanishing_count"], step)
        trainer.writer.add_scalar("monitor/exploding_count", stats["exploding_count"], step)

        # Check for gradient issues
        if stats["vanishing_count"] > 0:
            print(
                f"  ⚠️  WARNING (step {step}): "
                f"{int(stats['vanishing_count'])} vanishing gradients detected!"
            )

        if stats["exploding_count"] > 0:
            print(
                f"  ⚠️  WARNING (step {step}): "
                f"{int(stats['exploding_count'])} exploding gradients detected!"
            )

    def _log_update_stats(self, trainer: 'Trainer') -> None:
        """Log weight update statistics to TensorBoard.

        Args:
            trainer: The Trainer instance.
        """
        if self.current_update_diag is None:
            return

        step = trainer.global_step
        stats = WeightUpdateMonitor.aggregate_update_stats(self.current_update_diag)

        if stats.get("count", 0.0) == 0.0:
            return

        # Log aggregate statistics
        trainer.writer.add_scalar("monitor/update_ratio_median", stats["median"], step)
        trainer.writer.add_scalar("monitor/update_ratio_p95", stats["p95"], step)
        trainer.writer.add_scalar("monitor/update_ratio_min", stats["min"], step)
        trainer.writer.add_scalar("monitor/update_ratio_max", stats["max"], step)
        trainer.writer.add_scalar("monitor/frozen_count", stats["frozen_count"], step)

        # Log top-K worst (smallest update ratios)
        for name, val in WeightUpdateMonitor.top_k_smallest_updates(
            self.current_update_diag, k=self.monitor_topk
        ):
            trainer.writer.add_scalar(f"monitor/topk_frozen/{name}", val, step)

        # Check for frozen parameters
        if stats["frozen_count"] > 0:
            frozen_params = [
                name for name, diag in self.current_update_diag.items() if diag.is_frozen
            ]
            print(
                f"  ⚠️  WARNING (step {step}): "
                f"{int(stats['frozen_count'])} frozen parameters detected!"
            )
            print(f"     Frozen: {', '.join(frozen_params[:5])}")
            if len(frozen_params) > 5:
                print(f"     ... and {len(frozen_params) - 5} more")
