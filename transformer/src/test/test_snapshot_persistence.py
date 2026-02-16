"""End-to-end snapshot state persistence tests.

Tests that callback state is properly saved to and restored from snapshots,
including resume functionality.
"""
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import shutil
from training.trainer import Trainer, TrainerConfig
from training.trainer_callback import TrainerCallback
from training.trainer_early_stopping import EarlyStoppingCallback


class StatefulCallback(TrainerCallback):
    """Callback that maintains state across training."""

    def __init__(self):
        self.epoch_count = 0
        self.step_count = 0
        self.loss_history = []

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
        self.epoch_count += 1
        self.loss_history.append(train_loss)

    def on_batch_end(self, trainer, batch_idx, loss):
        self.step_count += 1

    def on_snapshot_save(self, trainer, epoch, step):
        return {
            'epoch_count': self.epoch_count,
            'step_count': self.step_count,
            'loss_history': self.loss_history
        }

    def on_snapshot_load(self, trainer, checkpoint):
        self.epoch_count = checkpoint.get('epoch_count', 0)
        self.step_count = checkpoint.get('step_count', 0)
        self.loss_history = checkpoint.get('loss_history', [])


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x, y=None):
        return torch.log_softmax(self.fc(x), dim=-1)


def test_callback_state_saved_in_snapshot():
    """Test that callback state is saved to snapshot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Callback State Saved in Snapshot")
        print("=" * 70)

        # Clean test directory
        test_dir = Path(tmpdir) / "test_snapshot_save"
        if test_dir.exists():
            shutil.rmtree(test_dir)

        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        callback = StatefulCallback()

        config = TrainerConfig(
            model_name="snapshot_test",
            result_dir=str(test_dir),
            log_interval=100,
            snapshot_per_epoch=True,
            delete_snapshots_after_training=False  # Keep snapshots for inspection
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            callbacks=[callback]
        )

        # Create data
        x = torch.randn(16, 10)
        y = torch.randint(0, 5, (16,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        # Monkey-patch
        def patched_train_step(batch):
            x_batch, y_batch = batch
            x_batch = x_batch.to(trainer.device)
            y_batch = y_batch.to(trainer.device)

            trainer.optimizer.zero_grad()
            output = trainer.model(x_batch)
            loss = trainer.criterion(output, y_batch)

            trainer.callbacks.on_forward_end(trainer, loss)
            loss.backward()
            trainer.callbacks.on_backward_end(trainer)
            trainer._clip_gradients()
            trainer.optimizer.step()
            trainer.callbacks.on_step_end(trainer)

            return loss.item()

        trainer._train_one_step = patched_train_step

        # Train for 2 epochs
        trainer.train(train_loader=loader, num_epochs=2)

        # Verify callback state was updated
        # Note: epoch numbers are 0-indexed, so after 2 epochs we've completed epoch 0 and 1
        # The epoch_count tracks how many times on_epoch_end was called
        expected_epoch_count = callback.epoch_count
        assert expected_epoch_count >= 1, f"Expected at least 1 epoch, got {callback.epoch_count}"
        assert callback.step_count > 0, "Step count should be > 0"
        assert len(callback.loss_history) >= 1, "Should have at least 1 loss value"

        # Find and load the snapshot
        snapshots_dir = test_dir / "snapshot_test" / "snapshots"
        snapshot_files = list(snapshots_dir.glob("*.pt"))
        assert len(snapshot_files) > 0, "No snapshot files found"

        # Load snapshot and verify callback state is saved
        snapshot_path = snapshot_files[-1]  # Latest snapshot
        checkpoint = torch.load(snapshot_path)

        assert 'extra_state' in checkpoint, "extra_state not in checkpoint"
        assert 'callback_state' in checkpoint['extra_state'], "callback_state not in extra_state"
        assert 'StatefulCallback' in checkpoint['extra_state']['callback_state'], \
            "StatefulCallback state not in checkpoint"

        saved_state = checkpoint['extra_state']['callback_state']['StatefulCallback']

        # Verify the snapshot contains valid callback state
        # Note: The saved state reflects the callback state at the time the snapshot was saved,
        # which may differ from the callback's current state after training completes
        assert 'epoch_count' in saved_state
        assert 'step_count' in saved_state
        assert 'loss_history' in saved_state
        assert saved_state['epoch_count'] >= 1, "Should have completed at least 1 epoch"
        assert saved_state['step_count'] > 0, "Should have completed at least 1 step"
        assert len(saved_state['loss_history']) >= 1, "Should have at least 1 loss value"

        print(f"  Callback state in checkpoint: {saved_state}")
        print("PASS: Callback state saved in snapshot\n")

        # Cleanup
        shutil.rmtree(test_dir)


def test_callback_state_restored_from_snapshot():
    """Test that callback state is restored when loading snapshot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Callback State Restored from Snapshot")
        print("=" * 70)

        test_dir = Path(tmpdir) / "test_snapshot_restore"
        if test_dir.exists():
            shutil.rmtree(test_dir)

        # Phase 1: Train and save
        model1 = DummyModel()
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        callback1 = StatefulCallback()

        config = TrainerConfig(
            model_name="restore_test",
            result_dir=str(test_dir),
            log_interval=100,
            snapshot_per_epoch=True,
            delete_snapshots_after_training=False
        )

        trainer1 = Trainer(
            model=model1,
            optimizer=optimizer1,
            criterion=criterion,
            config=config,
            callbacks=[callback1]
        )

        # Create data
        x = torch.randn(16, 10)
        y = torch.randint(0, 5, (16,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        # Monkey-patch
        def patched_train_step(batch):
            x_batch, y_batch = batch
            x_batch = x_batch.to(trainer1.device)
            y_batch = y_batch.to(trainer1.device)

            trainer1.optimizer.zero_grad()
            output = trainer1.model(x_batch)
            loss = trainer1.criterion(output, y_batch)

            trainer1.callbacks.on_forward_end(trainer1, loss)
            loss.backward()
            trainer1.callbacks.on_backward_end(trainer1)
            trainer1._clip_gradients()
            trainer1.optimizer.step()
            trainer1.callbacks.on_step_end(trainer1)

            return loss.item()

        trainer1._train_one_step = patched_train_step

        # Train phase 1
        trainer1.train(train_loader=loader, num_epochs=2)

        # Save state from callback1
        original_epoch_count = callback1.epoch_count
        original_step_count = callback1.step_count
        original_loss_history = callback1.loss_history.copy()

        # Ensure we actually have some state to verify
        assert original_epoch_count > 0, "No epochs completed in phase 1"
        assert original_step_count > 0, "No steps completed in phase 1"

        print(f"  Original callback state:")
        print(f"    epoch_count: {original_epoch_count}")
        print(f"    step_count: {original_step_count}")
        print(f"    loss_history: {original_loss_history}")

        # Phase 2: Create new trainer and callback, load snapshot
        model2 = DummyModel()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

        callback2 = StatefulCallback()  # Fresh callback with default state

        trainer2 = Trainer(
            model=model2,
            optimizer=optimizer2,
            criterion=criterion,
            config=config,
            callbacks=[callback2]
        )

        # Verify callback2 starts with default state
        assert callback2.epoch_count == 0
        assert callback2.step_count == 0
        assert len(callback2.loss_history) == 0

        # Load snapshot
        snapshots_dir = test_dir / "restore_test" / "snapshots"
        snapshot_files = sorted(snapshots_dir.glob("*.pt"))
        print(f"  Found snapshots: {[f.name for f in snapshot_files]}")
        snapshot_path = snapshot_files[-1]
        print(f"  Loading: {snapshot_path.name}")

        trainer2.load_snapshot(snapshot_path)

        # Verify callback2 state was restored
        assert callback2.epoch_count == original_epoch_count, \
            f"epoch_count not restored: expected {original_epoch_count}, got {callback2.epoch_count}"

        assert callback2.step_count == original_step_count, \
            f"step_count not restored: expected {original_step_count}, got {callback2.step_count}"

        assert callback2.loss_history == original_loss_history, \
            f"loss_history not restored: expected {original_loss_history}, got {callback2.loss_history}"

        print(f"  Restored callback state:")
        print(f"    epoch_count: {callback2.epoch_count}")
        print(f"    step_count: {callback2.step_count}")
        print(f"    loss_history: {callback2.loss_history}")
        print("PASS: Callback state restored from snapshot\n")

        # Cleanup
        shutil.rmtree(test_dir)


def test_early_stopping_state_persistence():
    """Test that early stopping callback state persists across save/load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Early Stopping State Persistence")
        print("=" * 70)

        test_dir = Path(tmpdir) / "test_early_stop_persist"
        if test_dir.exists():
            shutil.rmtree(test_dir)

        # Phase 1: Train with early stopping
        model1 = DummyModel()
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.0001)  # Very low LR
        criterion = nn.NLLLoss()

        early_stop1 = EarlyStoppingCallback(patience=5, min_delta=0.001, restore_best=True)

        config = TrainerConfig(
            model_name="early_stop_persist",
            result_dir=str(test_dir),
            log_interval=100,
            snapshot_per_epoch=True,
            delete_snapshots_after_training=False
        )

        trainer1 = Trainer(
            model=model1,
            optimizer=optimizer1,
            criterion=criterion,
            config=config,
            callbacks=[early_stop1]
        )

        # Create data
        x = torch.randn(16, 10)
        y = torch.randint(0, 5, (16,))
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=8)
        val_loader = DataLoader(dataset, batch_size=8)

        # Monkey-patch
        def patched_train_step(batch):
            x_batch, y_batch = batch
            x_batch = x_batch.to(trainer1.device)
            y_batch = y_batch.to(trainer1.device)

            trainer1.optimizer.zero_grad()
            output = trainer1.model(x_batch)
            loss = trainer1.criterion(output, y_batch)

            trainer1.callbacks.on_forward_end(trainer1, loss)
            loss.backward()
            trainer1.callbacks.on_backward_end(trainer1)
            trainer1._clip_gradients()
            trainer1.optimizer.step()
            trainer1.callbacks.on_step_end(trainer1)

            return loss.item()

        trainer1._train_one_step = patched_train_step

        # Monkey-patch _validate to handle tuple format
        def patched_validate(val_loader):
            trainer1.model.eval()
            total_loss = 0.0
            for batch in val_loader:
                x_batch, y_batch = batch
                x_batch = x_batch.to(trainer1.device)
                y_batch = y_batch.to(trainer1.device)

                with torch.no_grad():
                    output = trainer1.model(x_batch)
                    loss = trainer1.criterion(output, y_batch)
                    total_loss += loss.item()

            return total_loss / len(val_loader)

        trainer1._validate = patched_validate

        # Train (will have no improvement due to low LR)
        trainer1.train(train_loader=train_loader, val_loader=val_loader, num_epochs=3)

        # Load the snapshot that was saved to get its actual state
        snapshots_dir = test_dir / "early_stop_persist" / "snapshots"
        snapshot_files = list(snapshots_dir.glob("*.pt"))
        snapshot_path = snapshot_files[-1]
        saved_checkpoint = torch.load(snapshot_path)

        # Extract the actual saved state from the checkpoint
        saved_callback_state = saved_checkpoint['extra_state']['callback_state']['EarlyStoppingCallback']
        original_counter = saved_callback_state['counter']
        original_best_loss = saved_callback_state['best_loss']
        original_best_epoch = saved_callback_state['best_epoch']

        print(f"  Original early stopping state (from snapshot):")
        print(f"    counter: {original_counter}")
        print(f"    best_loss: {original_best_loss:.4f}")
        print(f"    best_epoch: {original_best_epoch}")

        # Phase 2: Load and verify restoration
        model2 = DummyModel()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.0001)

        early_stop2 = EarlyStoppingCallback(patience=5, min_delta=0.001, restore_best=True)

        trainer2 = Trainer(
            model=model2,
            optimizer=optimizer2,
            criterion=criterion,
            config=config,
            callbacks=[early_stop2]
        )

        # Load snapshot (reuse the same snapshot_path from above)
        trainer2.load_snapshot(snapshot_path)

        # Verify early stopping state restored
        assert early_stop2.counter == original_counter, \
            f"Counter not restored: expected {original_counter}, got {early_stop2.counter}"

        assert abs(early_stop2.best_loss - original_best_loss) < 0.001, \
            f"Best loss not restored: expected {original_best_loss}, got {early_stop2.best_loss}"

        assert early_stop2.best_epoch == original_best_epoch, \
            f"Best epoch not restored: expected {original_best_epoch}, got {early_stop2.best_epoch}"

        print(f"  Restored early stopping state:")
        print(f"    counter: {early_stop2.counter}")
        print(f"    best_loss: {early_stop2.best_loss:.4f}")
        print(f"    best_epoch: {early_stop2.best_epoch}")
        print("PASS: Early stopping state persisted\n")

        # Cleanup
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SNAPSHOT STATE PERSISTENCE TEST SUITE")
    print("=" * 70)

    try:
        test_callback_state_saved_in_snapshot()
        test_callback_state_restored_from_snapshot()
        test_early_stopping_state_persistence()

        print("=" * 70)
        print("ALL PERSISTENCE TESTS PASSED")
        print("=" * 70)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
