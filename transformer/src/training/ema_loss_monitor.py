"""Exponential Moving Average (EMA) loss monitoring for training.

This module provides EMA loss tracking to smooth noisy step-by-step loss values
and reveal underlying learning trends.

Usage:
    from training.ema_loss_monitor import EMALossMonitor

    monitor = EMALossMonitor(alpha=0.99)

    # During training loop
    for step, loss in enumerate(losses):
        ema = monitor.update(loss)
        writer.add_scalar("train/loss_ema", ema, step)

    # Checkpoint management
    checkpoint['ema_state'] = monitor.state_dict()
    monitor.load_state_dict(checkpoint['ema_state'])
"""
from typing import Optional


class EMALossMonitor:
    """Exponential Moving Average (EMA) loss monitor.

    Provides smoothed loss visualization by computing exponential moving average
    of training loss. The EMA reduces noise from batch-to-batch variance while
    revealing the underlying learning trend.

    Mathematical Definition:
        ema(t) = α × loss(t) + (1 - α) × ema(t-1)

    Where α (alpha) is the decay factor. Higher alpha gives more weight to recent
    values, lower alpha provides more smoothing.

    Typical values:
        - α = 0.99: Smooths over ~100 steps (1/(1-α))
        - α = 0.95: Smooths over ~20 steps
        - α = 0.999: Smooths over ~1000 steps

    Example:
        Without EMA (raw loss):
            Step 100: 2.45
            Step 101: 2.12  ← good batch
            Step 102: 2.89  ← hard batch
            Step 103: 2.01
            Trend: Unclear, noisy

        With EMA (α=0.99):
            Step 100: 2.45
            Step 101: 2.42  ← smooth downward
            Step 102: 2.43  ← slight up
            Step 103: 2.39  ← continuing down
            Trend: Clear decreasing

    Attributes:
        alpha: EMA decay factor (0 < alpha < 1).
        ema_value: Current EMA value (None until first update).
    """

    def __init__(self, alpha: float = 0.99):
        """Initialize EMA loss monitor.

        Args:
            alpha: EMA decay factor (0 < alpha < 1). Higher values give more
                weight to recent losses. Typical: 0.99 for ~100-step smoothing.

        Raises:
            ValueError: If alpha is not in (0, 1).
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.ema_value: Optional[float] = None

    def update(self, loss: float) -> float:
        """Update EMA with new loss value and return current EMA.

        Args:
            loss: Current step loss value.

        Returns:
            Updated EMA loss value.
        """
        if self.ema_value is None:
            # Cold start: Initialize EMA with first loss value
            self.ema_value = loss
        else:
            # Update: ema = alpha * loss + (1 - alpha) * ema
            self.ema_value = self.alpha * loss + (1 - self.alpha) * self.ema_value

        return self.ema_value

    def get(self) -> Optional[float]:
        """Get current EMA value without updating.

        Returns:
            Current EMA value, or None if not yet initialized.
        """
        return self.ema_value

    def reset(self) -> None:
        """Reset EMA monitor to initial state."""
        self.ema_value = None

    def state_dict(self) -> dict:
        """Return state dictionary for checkpoint saving.

        Returns:
            Dictionary containing EMA state.
        """
        return {
            'alpha': self.alpha,
            'ema_value': self.ema_value
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint.

        Args:
            state: State dictionary from state_dict().
        """
        self.alpha = state['alpha']
        self.ema_value = state.get('ema_value', None)

    def __repr__(self) -> str:
        """String representation of monitor."""
        return f"EMALossMonitor(alpha={self.alpha}, ema_value={self.ema_value})"


if __name__ == "__main__":
    print(__doc__)
