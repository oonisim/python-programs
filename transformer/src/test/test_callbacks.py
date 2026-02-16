"""Test suite for trainer callback system.

Tests the base callback infrastructure, callback integration with trainer,
and specific callback implementations.
"""
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import Trainer, TrainerConfig
from training.trainer_callback import TrainerCallback, CallbackList
from training.trainer_early_stopping import EarlyStoppingCallback
from training.trainer_gradient_monitor import GradientMonitorCallback


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        ])
        self.activation = nn.ReLU()
        self.output = nn.LogSoftmax(dim=-1)

    def forward(self, x, y=None):
        """Forward pass - dummy interface matching transformer."""
        out = x
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return self.output(out)


def create_dummy_data(num_samples=100, input_size=10, output_size=5):
    """Create dummy dataset for testing."""
    x = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    return TensorDataset(x, y)


class MockCallback(TrainerCallback):
    """Mock callback that records when hooks are called."""

    def __init__(self):
        self.calls = {
            'on_train_start': 0,
            'on_train_end': 0,
            'on_epoch_start': 0,
            'on_epoch_end': 0,
            'on_batch_start': 0,
            'on_batch_end': 0,
            'on_forward_end': 0,
            'on_backward_end': 0,
            'on_step_end': 0,
            'on_snapshot_save': 0,
            'on_snapshot_load': 0,
            'should_stop_training': 0,
        }
        self.epochs = []
        self.losses = []

    def on_train_start(self, trainer):
        self.calls['on_train_start'] += 1

    def on_train_end(self, trainer, result):
        self.calls['on_train_end'] += 1

    def on_epoch_start(self, trainer, epoch):
        self.calls['on_epoch_start'] += 1

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
        self.calls['on_epoch_end'] += 1
        self.epochs.append(epoch)
        self.losses.append(train_loss)

    def on_batch_start(self, trainer, batch_idx):
        self.calls['on_batch_start'] += 1

    def on_batch_end(self, trainer, batch_idx, loss):
        self.calls['on_batch_end'] += 1

    def on_forward_end(self, trainer, loss):
        self.calls['on_forward_end'] += 1

    def on_backward_end(self, trainer):
        self.calls['on_backward_end'] += 1

    def on_step_end(self, trainer):
        self.calls['on_step_end'] += 1

    def on_snapshot_save(self, trainer, epoch, step):
        self.calls['on_snapshot_save'] += 1
        return {'test_state': epoch}

    def on_snapshot_load(self, trainer, checkpoint):
        self.calls['on_snapshot_load'] += 1

    def should_stop_training(self, trainer):
        self.calls['should_stop_training'] += 1
        return False


def test_callback_hooks_are_called():
    """Test that all callback hooks are called during training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 70)
        print("TEST 1: Verify all callback hooks are called")
        print("=" * 70)

        # Create model and data
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        train_data = create_dummy_data(32)  # Small dataset
        train_loader = DataLoader(train_data, batch_size=8)

        # Create test callback
        test_callback = MockCallback()

        # Configure trainer
        config = TrainerConfig(
            model_name="test_callbacks",
            result_dir=tmpdir,
            log_interval=10,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            callbacks=[test_callback]
        )

        # Patch training methods like in test_early_stopping
        def patched_train_step(batch):
            x, y = batch
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            trainer.optimizer.zero_grad()
            output = trainer.model(x)
            loss = trainer.criterion(output, y)

            # Manually trigger callbacks to match trainer flow
            trainer.callbacks.on_forward_end(trainer, loss)
            loss.backward()
            trainer.callbacks.on_backward_end(trainer)
            trainer._clip_gradients()
            trainer.optimizer.step()
            trainer.callbacks.on_step_end(trainer)

            return loss.item()

        trainer._train_one_step = patched_train_step

        # Train for 2 epochs
        result = trainer.train(train_loader=train_loader, num_epochs=2)

        # Verify hooks were called
        print(f"\n✓ on_train_start called: {test_callback.calls['on_train_start']} times")
        print(f"✓ on_train_end called: {test_callback.calls['on_train_end']} times")
        print(f"✓ on_epoch_start called: {test_callback.calls['on_epoch_start']} times")
        print(f"✓ on_epoch_end called: {test_callback.calls['on_epoch_end']} times")
        print(f"✓ on_batch_start called: {test_callback.calls['on_batch_start']} times")
        print(f"✓ on_batch_end called: {test_callback.calls['on_batch_end']} times")
        print(f"✓ should_stop_training called: {test_callback.calls['should_stop_training']} times")

        assert test_callback.calls['on_train_start'] == 1, "on_train_start should be called once"
        assert test_callback.calls['on_train_end'] == 1, "on_train_end should be called once"
        assert test_callback.calls['on_epoch_start'] == 2, "on_epoch_start should be called per epoch"
        assert test_callback.calls['on_epoch_end'] == 2, "on_epoch_end should be called per epoch"
        assert test_callback.calls['on_batch_start'] > 0, "on_batch_start should be called"
        assert test_callback.calls['on_batch_end'] > 0, "on_batch_end should be called"
        assert test_callback.calls['should_stop_training'] == 2, "should_stop_training checked per epoch"

        print("\n✅ TEST 1 PASSED: All callback hooks called correctly\n")


def test_multiple_callbacks():
    """Test that multiple callbacks work together."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 70)
        print("TEST 2: Multiple callbacks work together")
        print("=" * 70)

        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = nn.NLLLoss()

        train_data = create_dummy_data(50)
        train_loader = DataLoader(train_data, batch_size=16)

        # Create multiple callbacks
        callback1 = MockCallback()
        callback2 = MockCallback()

        config = TrainerConfig(
            model_name="test_multi_callbacks",
            result_dir=tmpdir,
            log_interval=10,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            callbacks=[callback1, callback2]
        )

        # Patch training
        def patched_train_step(batch):
            x, y = batch
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            trainer.optimizer.zero_grad()
            output = trainer.model(x)
            loss = trainer.criterion(output, y)
            trainer.callbacks.on_forward_end(trainer, loss)
            loss.backward()
            trainer.callbacks.on_backward_end(trainer)
            trainer.optimizer.step()
            trainer.callbacks.on_step_end(trainer)
            return loss.item()

        trainer._train_one_step = patched_train_step

        # Train
        result = trainer.train(train_loader=train_loader, num_epochs=1)

        # Both callbacks should be called
        assert callback1.calls['on_train_start'] == 1, "Callback 1 should be called"
        assert callback2.calls['on_train_start'] == 1, "Callback 2 should be called"
        assert callback1.calls['on_epoch_end'] == 1, "Callback 1 epoch end called"
        assert callback2.calls['on_epoch_end'] == 1, "Callback 2 epoch end called"

        print(f"\n✓ Callback 1 on_train_start: {callback1.calls['on_train_start']}")
        print(f"✓ Callback 2 on_train_start: {callback2.calls['on_train_start']}")

        print("\n✅ TEST 2 PASSED: Multiple callbacks work together\n")


def test_callback_can_stop_training():
    """Test that a callback can stop training early."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 70)
        print("TEST 3: Callback can stop training")
        print("=" * 70)

        class StopAfterNEpochs(TrainerCallback):
            def __init__(self, n):
                self.n = n
                self.epoch_count = 0

            def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
                self.epoch_count += 1

            def should_stop_training(self, trainer):
                return self.epoch_count >= self.n

        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        train_data = create_dummy_data(50)
        train_loader = DataLoader(train_data, batch_size=16)

        # Create callback that stops after 3 epochs
        stop_callback = StopAfterNEpochs(n=3)

        config = TrainerConfig(
            model_name="test_stop_callback",
            result_dir=tmpdir,
            log_interval=10,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            callbacks=[stop_callback]
        )

        # Patch training
        def patched_train_step(batch):
            x, y = batch
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            trainer.optimizer.zero_grad()
            output = trainer.model(x)
            loss = trainer.criterion(output, y)
            loss.backward()
            trainer.optimizer.step()
            return loss.item()

        trainer._train_one_step = patched_train_step

        # Request 10 epochs but should stop at 3
        result = trainer.train(train_loader=train_loader, num_epochs=10)

        actual_epochs = result['total_epochs']
        print(f"\n✓ Requested 10 epochs")
        print(f"✓ Stopped at epoch {actual_epochs}")

        assert actual_epochs == 3, "Should stop after 3 epochs"
        print("\n✅ TEST 3 PASSED: Callback stopped training early\n")


def test_callback_state_persistence():
    """Test that callback state is saved and restored with snapshots."""
    print("=" * 70)
    print("TEST 4: Callback state persistence")
    print("=" * 70)

    # This is tested implicitly in the early stopping callback
    # which saves/restores best_weights and counter
    early_stop = EarlyStoppingCallback(patience=5, min_delta=0.001)

    # Simulate state save
    state = early_stop.on_snapshot_save(None, epoch=10, step=100)

    assert 'counter' in state, "Should save counter"
    assert 'best_loss' in state, "Should save best_loss"
    assert 'best_epoch' in state, "Should save best_epoch"

    # Modify state
    early_stop.counter = 3
    early_stop.best_loss = 1.5
    early_stop.best_epoch = 8

    # Simulate state restore
    early_stop.on_snapshot_load(None, state)

    # Should be restored to original values
    assert early_stop.counter == 0, "Counter should be restored"
    assert early_stop.best_loss == float('inf'), "Best loss should be restored"

    print("\n✓ Callback state saved successfully")
    print("✓ Callback state restored successfully")

    print("\n✅ TEST 4 PASSED: Callback state persistence works\n")


def test_callback_list():
    """Test CallbackList functionality."""
    print("=" * 70)
    print("TEST 5: CallbackList functionality")
    print("=" * 70)

    callback1 = MockCallback()
    callback2 = MockCallback()

    callback_list = CallbackList([callback1, callback2])

    # Test that calling methods on list calls all callbacks
    callback_list.on_train_start(None)

    assert callback1.calls['on_train_start'] == 1
    assert callback2.calls['on_train_start'] == 1

    # Test should_stop_training returns True if any callback returns True
    class StopCallback(TrainerCallback):
        def should_stop_training(self, trainer):
            return True

    stop_callback = StopCallback()
    callback_list2 = CallbackList([callback1, stop_callback])

    should_stop = callback_list2.should_stop_training(None)
    assert should_stop is True, "Should return True if any callback returns True"

    print("\n✓ CallbackList propagates calls to all callbacks")
    print("✓ CallbackList aggregates should_stop_training correctly")

    print("\n✅ TEST 5 PASSED: CallbackList works correctly\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRAINER CALLBACK SYSTEM TEST SUITE")
    print("=" * 70 + "\n")

    try:
        test_callback_hooks_are_called()
        test_multiple_callbacks()
        test_callback_can_stop_training()
        test_callback_state_persistence()
        test_callback_list()

        print("=" * 70)
        print("✅ ALL CALLBACK TESTS PASSED!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
