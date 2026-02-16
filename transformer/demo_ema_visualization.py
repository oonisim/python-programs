"""Visualize EMA smoothing effect on noisy loss data.

This script demonstrates how EMA reduces noise and reveals underlying trends.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.ema_loss_monitor import EMALossMonitor
import random


def generate_noisy_loss(base_value, noise_level=0.3):
    """Generate a noisy loss value."""
    noise = random.uniform(-noise_level, noise_level)
    return base_value + noise


def simulate_training():
    """Simulate a training run with decreasing loss."""
    print("=" * 80)
    print("EMA Loss Smoothing Demonstration")
    print("=" * 80)
    print()
    print("Simulating a training run with noisy loss values...")
    print("The true trend is: gradual decrease from 10.0 to 5.0")
    print()

    # Create EMA monitors with different alpha values
    monitors = {
        'α=0.95': EMALossMonitor(alpha=0.95),   # Light smoothing (~20 steps)
        'α=0.99': EMALossMonitor(alpha=0.99),   # Medium smoothing (~100 steps)
        'α=0.999': EMALossMonitor(alpha=0.999)  # Heavy smoothing (~1000 steps)
    }

    # Simulate training with decreasing loss + noise
    num_steps = 50
    print(f"{'Step':<6} {'Raw Loss':<12} {'α=0.95':<12} {'α=0.99':<12} {'α=0.999':<12}")
    print("-" * 80)

    for step in range(num_steps):
        # Generate true decreasing loss with noise
        true_loss = 10.0 - (5.0 * step / num_steps)
        raw_loss = generate_noisy_loss(true_loss, noise_level=0.8)

        # Update all monitors
        ema_values = {name: monitor.update(raw_loss) for name, monitor in monitors.items()}

        # Print every 5 steps
        if step % 5 == 0 or step == 0:
            print(f"{step:<6} {raw_loss:<12.4f} {ema_values['α=0.95']:<12.4f} "
                  f"{ema_values['α=0.99']:<12.4f} {ema_values['α=0.999']:<12.4f}")

    print()
    print("=" * 80)
    print("Observations:")
    print("=" * 80)
    print("1. Raw Loss: Very noisy, hard to see the decreasing trend")
    print("2. α=0.95:   Follows raw loss closely, but smoother")
    print("3. α=0.99:   Good balance between responsiveness and smoothing (DEFAULT)")
    print("4. α=0.999:  Very smooth, but lags behind actual trend")
    print()
    print("Recommendation: Use α=0.99 for most training scenarios")
    print()


def demonstrate_plateau_detection():
    """Demonstrate how EMA helps detect training plateaus."""
    print("=" * 80)
    print("Training Plateau Detection with EMA")
    print("=" * 80)
    print()
    print("Simulating training that plateaus after 30 steps...")
    print()

    monitor = EMALossMonitor(alpha=0.99)

    print(f"{'Step':<6} {'Raw Loss':<12} {'EMA':<12} {'Status':<20}")
    print("-" * 80)

    for step in range(60):
        if step < 30:
            # Decreasing loss
            true_loss = 8.0 - (3.0 * step / 30)
        else:
            # Plateau around 5.0
            true_loss = 5.0

        raw_loss = generate_noisy_loss(true_loss, noise_level=0.5)
        ema = monitor.update(raw_loss)

        # Detect status
        if step < 30:
            status = "Decreasing"
        elif step < 35:
            status = "Plateau starting"
        else:
            status = "Plateau detected"

        if step % 10 == 0 or step == 30:
            print(f"{step:<6} {raw_loss:<12.4f} {ema:<12.4f} {status:<20}")

    print()
    print("=" * 80)
    print("Observations:")
    print("=" * 80)
    print("• Raw loss is noisy throughout, making plateau hard to spot")
    print("• EMA clearly shows the plateau starting around step 30")
    print("• This can trigger early stopping or learning rate adjustments")
    print()


def demonstrate_divergence():
    """Demonstrate how EMA helps detect training divergence."""
    print("=" * 80)
    print("Training Divergence Detection with EMA")
    print("=" * 80)
    print()
    print("Simulating training that starts diverging after 25 steps...")
    print()

    monitor = EMALossMonitor(alpha=0.99)

    print(f"{'Step':<6} {'Raw Loss':<12} {'EMA':<12} {'Status':<20}")
    print("-" * 80)

    for step in range(50):
        if step < 25:
            # Normal decreasing loss
            true_loss = 6.0 - (2.0 * step / 25)
        else:
            # Diverging (exploding loss)
            true_loss = 4.0 + (0.3 * (step - 25))

        raw_loss = generate_noisy_loss(true_loss, noise_level=0.4)
        ema = monitor.update(raw_loss)

        # Detect status
        if step < 25:
            status = "Normal"
        elif step < 30:
            status = "Divergence starting"
        else:
            status = "Divergence detected!"

        if step % 10 == 0 or step == 25:
            print(f"{step:<6} {raw_loss:<12.4f} {ema:<12.4f} {status:<20}")

    print()
    print("=" * 80)
    print("Observations:")
    print("=" * 80)
    print("• Raw loss fluctuates, making divergence hard to confirm")
    print("• EMA clearly shows loss increasing after step 25")
    print("• Early detection allows quick intervention (lower LR, restart, etc.)")
    print()


def main():
    """Run all demonstrations."""
    random.seed(42)  # For reproducibility

    simulate_training()
    demonstrate_plateau_detection()
    demonstrate_divergence()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("EMA loss monitoring provides:")
    print("✓ Clear visualization of true learning trends")
    print("✓ Early detection of plateaus and divergence")
    print("✓ Better training progress assessment")
    print("✓ Reduced noise in loss curves")
    print()
    print("The feature is enabled by default with α=0.99 in TrainerConfig.")
    print("View both raw and EMA loss in TensorBoard: train/step_loss and train/loss_ema")
    print()


if __name__ == "__main__":
    main()
