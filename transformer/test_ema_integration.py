"""Test script to verify EMA loss monitor integration.

This script tests:
1. EMA monitor initialization
2. EMA updates during training
3. EMA state saving and loading in checkpoints
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.trainer import TrainerConfig, LanguageModelTrainer
from training.ema_loss_monitor import EMALossMonitor


def test_ema_monitor_standalone():
    """Test EMALossMonitor class standalone."""
    print("=" * 70)
    print("Test 1: EMALossMonitor standalone")
    print("=" * 70)

    monitor = EMALossMonitor(alpha=0.9)

    # Simulate noisy loss values
    losses = [10.0, 9.5, 10.2, 9.8, 9.3, 9.7, 9.2, 9.0, 9.4, 8.9]

    print(f"Alpha: {monitor.alpha}")
    print(f"\n{'Step':<6} {'Loss':<8} {'EMA':<8}")
    print("-" * 25)

    for step, loss in enumerate(losses):
        ema = monitor.update(loss)
        print(f"{step:<6} {loss:<8.4f} {ema:<8.4f}")

    # Test state dict
    state = monitor.state_dict()
    print(f"\nState dict: {state}")

    # Test load state dict
    new_monitor = EMALossMonitor(alpha=0.95)
    new_monitor.load_state_dict(state)
    print(f"Loaded state: alpha={new_monitor.alpha}, ema_value={new_monitor.ema_value}")

    print("✓ EMALossMonitor standalone test passed\n")


def test_ema_in_trainer_config():
    """Test that trainer config includes EMA settings."""
    print("=" * 70)
    print("Test 2: TrainerConfig EMA settings")
    print("=" * 70)

    config = TrainerConfig(
        model_name="test_ema",
        enable_ema_loss=True,
        ema_alpha=0.99
    )

    print(f"enable_ema_loss: {config.enable_ema_loss}")
    print(f"ema_alpha: {config.ema_alpha}")

    assert hasattr(config, 'enable_ema_loss'), "Config missing enable_ema_loss"
    assert hasattr(config, 'ema_alpha'), "Config missing ema_alpha"

    print("✓ TrainerConfig EMA settings test passed\n")


def test_ema_monitor_initialization():
    """Test that trainer initializes EMA monitor correctly."""
    print("=" * 70)
    print("Test 3: Trainer EMA monitor initialization")
    print("=" * 70)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return torch.log_softmax(self.linear(x), dim=-1)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    config = TrainerConfig(
        model_name="test_ema",
        enable_ema_loss=True,
        ema_alpha=0.99,
        result_dir="test_results"
    )

    trainer = LanguageModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config
    )

    # Check that EMA monitor was initialized
    assert hasattr(trainer, 'ema_monitor'), "Trainer missing ema_monitor attribute"
    assert trainer.ema_monitor is not None, "EMA monitor not initialized"
    assert trainer.ema_monitor.alpha == 0.99, f"Expected alpha=0.99, got {trainer.ema_monitor.alpha}"

    print(f"EMA monitor initialized: {trainer.ema_monitor}")
    print("✓ Trainer EMA monitor initialization test passed\n")

    # Test with EMA disabled
    config_disabled = TrainerConfig(
        model_name="test_ema_disabled",
        enable_ema_loss=False,
        result_dir="test_results"
    )

    trainer_disabled = LanguageModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config_disabled
    )

    assert trainer_disabled.ema_monitor is None, "EMA monitor should be None when disabled"
    print("EMA monitor correctly disabled when enable_ema_loss=False")
    print("✓ EMA disable test passed\n")


def test_checkpoint_with_ema():
    """Test that EMA state is saved and loaded with checkpoints."""
    print("=" * 70)
    print("Test 4: Checkpoint EMA state save/load")
    print("=" * 70)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return torch.log_softmax(self.linear(x), dim=-1)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    config = TrainerConfig(
        model_name="test_ema_checkpoint",
        enable_ema_loss=True,
        ema_alpha=0.95,
        result_dir="test_results",
        snapshot_per_epoch=False  # Disable auto-snapshots
    )

    trainer = LanguageModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config
    )

    # Simulate some EMA updates
    for loss in [10.0, 9.5, 9.0, 8.5, 8.0]:
        trainer.ema_monitor.update(loss)

    ema_before = trainer.ema_monitor.get()
    print(f"EMA value before save: {ema_before}")

    # Save checkpoint
    snapshot_path = trainer.save_snapshot(epoch=0, step=100)
    print(f"Saved snapshot: {snapshot_path}")

    # Load checkpoint to verify state is saved
    checkpoint = torch.load(snapshot_path)
    assert 'ema_state' in checkpoint, "Checkpoint missing ema_state"
    print(f"EMA state in checkpoint: {checkpoint['ema_state']}")

    # Create new trainer and load checkpoint
    model2 = SimpleModel()
    optimizer2 = torch.optim.Adam(model2.parameters())

    trainer2 = LanguageModelTrainer(
        model=model2,
        optimizer=optimizer2,
        criterion=criterion,
        config=config
    )

    # Load checkpoint
    trainer2.load_snapshot(snapshot_path.name)
    ema_after = trainer2.ema_monitor.get()

    print(f"EMA value after load: {ema_after}")

    assert abs(ema_before - ema_after) < 1e-6, f"EMA mismatch: {ema_before} != {ema_after}"

    print("✓ Checkpoint EMA state save/load test passed\n")

    # Cleanup
    import shutil
    test_dir = Path("test_results")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("Cleaned up test directory")


def main():
    """Run all EMA integration tests."""
    print("\n")
    print("=" * 70)
    print("EMA Loss Monitor Integration Tests")
    print("=" * 70)
    print("\n")

    try:
        test_ema_monitor_standalone()
        test_ema_in_trainer_config()
        test_ema_monitor_initialization()
        test_checkpoint_with_ema()

        print("=" * 70)
        print("✓ All EMA integration tests passed!")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Test failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
