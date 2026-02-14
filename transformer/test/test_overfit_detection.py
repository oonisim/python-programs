"""Test early stopping with overfitting detection."""
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.trainer import Trainer, TrainerConfig
from training.trainer_early_stopping import EarlyStoppingCallback


class SimpleModel(nn.Module):
    """Simple linear model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x, y=None):
        return self.fc(x)


def test_overfit_detection_triggers():
    """Test that overfitting detection triggers when val-train gap keeps growing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Overfitting Detection Triggers")
        print("=" * 70)

        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Create callback with overfitting detection
        callback = EarlyStoppingCallback(
            patience=10,  # High patience for standard early stopping
            min_delta=0.001,
            restore_best=True,
            overfit_patience=3,  # Stop if gap grows for 3 epochs
            overfit_min_delta=0.05  # Minimum gap increase
        )

        config = TrainerConfig(
            model_name="test_overfit",
            result_dir=tmpdir,
            log_interval=100,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            callbacks=[callback]
        )

        # Create synthetic data where model will overfit
        # Train: easy pattern
        x_train = torch.randn(32, 10)
        y_train = x_train.mean(dim=1, keepdim=True)
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=8)

        # Val: different distribution (harder)
        x_val = torch.randn(32, 10) + 2.0  # Shifted distribution
        y_val = x_val.mean(dim=1, keepdim=True)
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=8)

        # Manually simulate training with growing gap
        class MockTrainer:
            def __init__(self):
                self.current_epoch = 0
                self.model = model

        mock_trainer = MockTrainer()

        # Simulate epochs where gap keeps growing
        # Epoch 0: train=1.0, val=1.2, gap=0.2
        callback.on_epoch_end(mock_trainer, epoch=0, train_loss=1.0, val_loss=1.2)
        assert callback.overfit_counter == 0, "First epoch should not increment counter"

        # Epoch 1: train=0.8, val=1.3, gap=0.5 (increase by 0.3 > 0.05)
        callback.on_epoch_end(mock_trainer, epoch=1, train_loss=0.8, val_loss=1.3)
        assert callback.overfit_counter == 1, f"Gap increased, counter should be 1, got {callback.overfit_counter}"

        # Epoch 2: train=0.6, val=1.4, gap=0.8 (increase by 0.3 > 0.05)
        callback.on_epoch_end(mock_trainer, epoch=2, train_loss=0.6, val_loss=1.4)
        assert callback.overfit_counter == 2, f"Gap increased again, counter should be 2, got {callback.overfit_counter}"

        # Epoch 3: train=0.4, val=1.5, gap=1.1 (increase by 0.3 > 0.05)
        # This should trigger early stopping
        callback.on_epoch_end(mock_trainer, epoch=3, train_loss=0.4, val_loss=1.5)
        assert callback.overfit_counter == 3, f"Counter should reach 3, got {callback.overfit_counter}"
        assert callback.should_stop_training(mock_trainer), "Should trigger early stopping"

        print("  ✓ Overfitting detection triggered after 3 epochs of growing gap")
        print("PASS: Overfitting detection works correctly\n")


def test_overfit_detection_resets_on_improvement():
    """Test that overfitting counter resets when gap stops growing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Overfitting Counter Resets")
        print("=" * 70)

        model = SimpleModel()

        callback = EarlyStoppingCallback(
            patience=10,
            min_delta=0.001,
            restore_best=True,
            overfit_patience=3,
            overfit_min_delta=0.05
        )

        class MockTrainer:
            def __init__(self):
                self.model = model

        mock_trainer = MockTrainer()

        # Epoch 0: gap=0.2
        callback.on_epoch_end(mock_trainer, epoch=0, train_loss=1.0, val_loss=1.2)
        assert callback.overfit_counter == 0

        # Epoch 1: gap=0.5 (increase by 0.3)
        callback.on_epoch_end(mock_trainer, epoch=1, train_loss=0.8, val_loss=1.3)
        assert callback.overfit_counter == 1

        # Epoch 2: gap=0.6 (increase by 0.1 > 0.05)
        callback.on_epoch_end(mock_trainer, epoch=2, train_loss=0.6, val_loss=1.2)
        assert callback.overfit_counter == 2

        # Epoch 3: gap=0.5 (DECREASE by 0.1) - should reset
        callback.on_epoch_end(mock_trainer, epoch=3, train_loss=0.5, val_loss=1.0)
        assert callback.overfit_counter == 0, \
            f"Counter should reset when gap decreases, got {callback.overfit_counter}"

        print("  ✓ Counter resets when gap stops growing")
        print("PASS: Counter reset logic works correctly\n")


def test_overfit_detection_disabled_by_default():
    """Test that overfitting detection is disabled when overfit_patience=0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Overfitting Detection Disabled")
        print("=" * 70)

        model = SimpleModel()

        callback = EarlyStoppingCallback(
            patience=5,
            min_delta=0.001,
            restore_best=True,
            overfit_patience=0  # Disabled
        )

        class MockTrainer:
            def __init__(self):
                self.model = model

        mock_trainer = MockTrainer()

        # Even with growing gap, should not trigger
        callback.on_epoch_end(mock_trainer, epoch=0, train_loss=1.0, val_loss=1.2)
        callback.on_epoch_end(mock_trainer, epoch=1, train_loss=0.8, val_loss=1.5)
        callback.on_epoch_end(mock_trainer, epoch=2, train_loss=0.6, val_loss=1.8)
        callback.on_epoch_end(mock_trainer, epoch=3, train_loss=0.4, val_loss=2.0)

        assert callback.overfit_counter == 0, "Counter should stay 0 when disabled"
        assert not callback.should_stop_training(mock_trainer), \
            "Should not stop when overfitting detection is disabled"

        print("  ✓ Overfitting detection remains disabled")
        print("PASS: Default behavior preserved\n")


if __name__ == "__main__":
    test_overfit_detection_triggers()
    test_overfit_detection_resets_on_improvement()
    test_overfit_detection_disabled_by_default()
    print("\nAll overfitting detection tests passed!")
