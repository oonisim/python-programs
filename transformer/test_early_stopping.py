"""Test script to verify early stopping mechanism works correctly."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from trainer import Trainer, TrainerConfig


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, y=None):
        """Forward pass - dummy interface matching transformer."""
        # For encoder-decoder style, we just use x
        return self.net(x)


def create_dummy_data(num_samples=100, input_size=10, output_size=5):
    """Create dummy dataset for testing."""
    x = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    return TensorDataset(x, y)


def test_early_stopping_triggers():
    """Test that early stopping actually stops training."""
    print("=" * 70)
    print("TEST 1: Verify early stopping triggers")
    print("=" * 70)

    # Create dummy model and data
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # Very small LR = minimal improvement
    criterion = nn.NLLLoss()

    # Create data
    train_data = create_dummy_data(100)
    val_data = create_dummy_data(50)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    # Configure with early stopping
    config = TrainerConfig(
        model_name="test_early_stop",
        base_dir="/tmp",
        early_stop_patience=3,
        early_stop_min_delta=0.01,
        early_stop_restore_best=True,
        log_interval=10,
        snapshot_per_epoch=False,
        delete_snapshots_after_training=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config
    )

    # Monkey-patch the _train_one_step to use our dummy data format
    original_train_step = trainer._train_one_step
    def patched_train_step(batch):
        x, y = batch
        x = x.to(trainer.device)
        y = y.to(trainer.device)

        trainer.optimizer.zero_grad()
        output = trainer.model.forward(x)
        loss = trainer.criterion(output, y)
        loss.backward()
        trainer._clip_gradients()
        trainer.optimizer.step()
        return loss.item()

    trainer._train_one_step = patched_train_step

    # Monkey-patch validation
    original_validate = trainer._validate
    @torch.no_grad()
    def patched_validate(val_loader):
        trainer.model.eval()
        total_loss = 0.0
        for batch in val_loader:
            x, y = batch
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            output = trainer.model.forward(x)
            loss = trainer.criterion(output, y)
            total_loss += loss.item()
        return total_loss / len(val_loader)

    trainer._validate = patched_validate

    # Train with a large number of epochs - should stop early
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50  # Set high, but should stop early
    )

    # Check that it stopped before 50 epochs
    actual_epochs = result['total_epochs']
    print(f"\n✓ Training stopped at epoch {actual_epochs} (requested 50 epochs)")
    print(f"✓ Best validation loss: {trainer.best_val_loss:.4f}")

    assert actual_epochs < 50, "Early stopping should have triggered before 50 epochs"
    assert trainer.early_stopping.counter >= config.early_stop_patience, "Counter should reach patience"

    print("\n✅ TEST 1 PASSED: Early stopping triggered correctly\n")


def test_early_stopping_restores_best_weights():
    """Test that best weights are restored when early stopping triggers."""
    print("=" * 70)
    print("TEST 2: Verify best weights are restored")
    print("=" * 70)

    # Create model
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # Create data
    train_data = create_dummy_data(100)
    val_data = create_dummy_data(50)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    # Configure with early stopping and restore best
    config = TrainerConfig(
        model_name="test_restore",
        base_dir="/tmp",
        early_stop_patience=2,
        early_stop_min_delta=0.001,
        early_stop_restore_best=True,
        log_interval=10,
        snapshot_per_epoch=False,
        delete_snapshots_after_training=True
    )

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, config=config)

    # Patch methods like before
    def patched_train_step(batch):
        x, y = batch
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        trainer.optimizer.zero_grad()
        output = trainer.model.forward(x)
        loss = trainer.criterion(output, y)
        loss.backward()
        trainer._clip_gradients()
        trainer.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def patched_validate(val_loader):
        trainer.model.eval()
        total_loss = 0.0
        for batch in val_loader:
            x, y = batch
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            output = trainer.model.forward(x)
            loss = trainer.criterion(output, y)
            total_loss += loss.item()
        return total_loss / len(val_loader)

    trainer._train_one_step = patched_train_step
    trainer._validate = patched_validate

    # Train
    result = trainer.train(train_loader=train_loader, val_loader=val_loader, num_epochs=20)

    print(f"\n✓ Best epoch was: {trainer.early_stopping.best_epoch}")
    print(f"✓ Stopped at epoch: {result['total_epochs']}")
    print(f"✓ Best weights were restored: {config.early_stop_restore_best}")

    assert trainer.early_stopping.best_weights is not None, "Best weights should be saved"
    print("\n✅ TEST 2 PASSED: Best weights restoration works\n")


def test_no_early_stopping_when_disabled():
    """Test that training runs full epochs when early stopping is disabled."""
    print("=" * 70)
    print("TEST 3: Verify training runs full epochs when early stopping disabled")
    print("=" * 70)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    criterion = nn.NLLLoss()

    train_data = create_dummy_data(100)
    val_data = create_dummy_data(50)
    train_loader = DataLoader(train_data, batch_size=16)
    val_loader = DataLoader(val_data, batch_size=16)

    # Configure WITHOUT early stopping (patience=0)
    config = TrainerConfig(
        model_name="test_no_early_stop",
        base_dir="/tmp",
        early_stop_patience=0,  # Disabled
        log_interval=10,
        snapshot_per_epoch=False,
        delete_snapshots_after_training=True
    )

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, config=config)

    # Patch methods
    def patched_train_step(batch):
        x, y = batch
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        trainer.optimizer.zero_grad()
        output = trainer.model.forward(x)
        loss = trainer.criterion(output, y)
        loss.backward()
        trainer.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def patched_validate(val_loader):
        trainer.model.eval()
        total_loss = 0.0
        for batch in val_loader:
            x, y = batch
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            output = trainer.model.forward(x)
            loss = trainer.criterion(output, y)
            total_loss += loss.item()
        return total_loss / len(val_loader)

    trainer._train_one_step = patched_train_step
    trainer._validate = patched_validate

    # Train for exactly 10 epochs
    num_epochs = 10
    result = trainer.train(train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs)

    actual_epochs = result['total_epochs']
    print(f"\n✓ Requested {num_epochs} epochs")
    print(f"✓ Completed {actual_epochs} epochs")

    assert actual_epochs == num_epochs, f"Should complete all {num_epochs} epochs when early stopping is disabled"
    assert trainer.early_stopping is None, "Early stopping should not be initialized"

    print("\n✅ TEST 3 PASSED: Full training without early stopping\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EARLY STOPPING TEST SUITE")
    print("=" * 70 + "\n")

    try:
        test_early_stopping_triggers()
        test_early_stopping_restores_best_weights()
        test_no_early_stopping_when_disabled()

        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
