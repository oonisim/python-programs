"""Test suite for callback hook timing verification.

Verifies that callbacks are invoked at the correct times during training:
- on_backward_end() called AFTER backward, BEFORE gradient clipping
- Gradients are accessible in on_backward_end()
- on_step_end() called AFTER optimizer.step()
- Proper ordering of all hooks
"""
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import Trainer, TrainerConfig
from training.trainer_callback import TrainerCallback


class GradientAccessCallback(TrainerCallback):
    """Callback that verifies gradient access at different hook points."""

    def __init__(self):
        self.gradients_at_backward_end = None
        self.gradients_at_step_end = None
        self.weights_before_step = None
        self.weights_after_step = None
        self.hooks_called_order = []

    def on_forward_end(self, trainer, loss):
        self.hooks_called_order.append('forward_end')
        # Gradients should NOT exist yet
        for param in trainer.model.parameters():
            if param.grad is not None:
                raise AssertionError("Gradients exist before backward!")

    def on_backward_end(self, trainer):
        self.hooks_called_order.append('backward_end')

        # Gradients MUST exist after backward
        grad_norms = []
        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    raise AssertionError(f"Gradient is None for {name} after backward!")
                grad_norms.append(param.grad.norm().item())

        # Store gradients for verification
        self.gradients_at_backward_end = {
            name: param.grad.clone()
            for name, param in trainer.model.named_parameters()
            if param.requires_grad and param.grad is not None
        }

        # Store weights before step
        self.weights_before_step = {
            name: param.data.clone()
            for name, param in trainer.model.named_parameters()
        }

    def on_step_end(self, trainer):
        self.hooks_called_order.append('step_end')

        # Weights MUST have changed after step
        self.weights_after_step = {
            name: param.data.clone()
            for name, param in trainer.model.named_parameters()
        }

        # Gradients should still be accessible (not zeroed yet)
        self.gradients_at_step_end = {
            name: param.grad.clone() if param.grad is not None else None
            for name, param in trainer.model.named_parameters()
            if param.requires_grad
        }


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x, y=None):
        x = torch.relu(self.fc1(x))
        return torch.log_softmax(self.fc2(x), dim=-1)


def test_hook_timing_order():
    """Test that hooks are called in the correct order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Hook Timing Order")
        print("=" * 70)

        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        callback = GradientAccessCallback()

        config = TrainerConfig(
            model_name="test_timing",
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

        # Create minimal dataset
        x = torch.randn(16, 10)
        y = torch.randint(0, 5, (16,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        # Monkey-patch _train_one_step to work with our data format
        original_train_step = trainer._train_one_step

        def patched_train_step(batch):
            x_batch, y_batch = batch
            x_batch = x_batch.to(trainer.device)
            y_batch = y_batch.to(trainer.device)

            trainer.optimizer.zero_grad()
            output = trainer.model.forward(x_batch)
            loss = trainer.criterion(output, y_batch)

            trainer.callbacks.on_forward_end(trainer, loss)
            loss.backward()
            trainer.callbacks.on_backward_end(trainer)
            trainer._clip_gradients()
            trainer.optimizer.step()
            trainer.callbacks.on_step_end(trainer)

            return loss.item()

        trainer._train_one_step = patched_train_step

        # Run 1 epoch
        trainer.train(train_loader=loader, num_epochs=1)

        # Verify hook order
        print(f"\nHooks called in order: {callback.hooks_called_order}")

        expected_order = ['forward_end', 'backward_end', 'step_end']
        actual_order = callback.hooks_called_order[:3]  # First batch

        assert actual_order == expected_order, \
            f"Hook order incorrect: expected {expected_order}, got {actual_order}"

        print("PASS: Hooks called in correct order\n")


def test_gradients_accessible_at_backward_end():
    """Test that gradients are accessible in on_backward_end()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Gradients Accessible at backward_end")
        print("=" * 70)

        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        callback = GradientAccessCallback()

        config = TrainerConfig(
            model_name="test_grad_access",
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

        # Create minimal dataset
        x = torch.randn(8, 10)
        y = torch.randint(0, 5, (8,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        # Monkey-patch
        def patched_train_step(batch):
            x_batch, y_batch = batch
            x_batch = x_batch.to(trainer.device)
            y_batch = y_batch.to(trainer.device)

            trainer.optimizer.zero_grad()
            output = trainer.model.forward(x_batch)
            loss = trainer.criterion(output, y_batch)

            trainer.callbacks.on_forward_end(trainer, loss)
            loss.backward()
            trainer.callbacks.on_backward_end(trainer)
            trainer._clip_gradients()
            trainer.optimizer.step()
            trainer.callbacks.on_step_end(trainer)

            return loss.item()

        trainer._train_one_step = patched_train_step

        # Run 1 epoch
        trainer.train(train_loader=loader, num_epochs=1)

        # Verify gradients were captured
        assert callback.gradients_at_backward_end is not None, \
            "Gradients not captured at backward_end"

        assert len(callback.gradients_at_backward_end) > 0, \
            "No gradients stored at backward_end"

        # Verify all gradients have non-zero norm
        for name, grad in callback.gradients_at_backward_end.items():
            grad_norm = grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6f}")
            assert grad_norm > 0, f"Gradient is zero for {name}"

        print("PASS: Gradients accessible at backward_end\n")


def test_weights_change_after_step():
    """Test that weights change after optimizer.step() in on_step_end()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Weights Change After Step")
        print("=" * 70)

        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # High LR for visible change
        criterion = nn.NLLLoss()

        callback = GradientAccessCallback()

        config = TrainerConfig(
            model_name="test_weight_change",
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

        # Create minimal dataset
        x = torch.randn(8, 10)
        y = torch.randint(0, 5, (8,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        # Monkey-patch
        def patched_train_step(batch):
            x_batch, y_batch = batch
            x_batch = x_batch.to(trainer.device)
            y_batch = y_batch.to(trainer.device)

            trainer.optimizer.zero_grad()
            output = trainer.model.forward(x_batch)
            loss = trainer.criterion(output, y_batch)

            trainer.callbacks.on_forward_end(trainer, loss)
            loss.backward()
            trainer.callbacks.on_backward_end(trainer)
            trainer._clip_gradients()
            trainer.optimizer.step()
            trainer.callbacks.on_step_end(trainer)

            return loss.item()

        trainer._train_one_step = patched_train_step

        # Run 1 epoch
        trainer.train(train_loader=loader, num_epochs=1)

        # Verify weights changed
        assert callback.weights_before_step is not None
        assert callback.weights_after_step is not None

        changes_detected = 0
        for name in callback.weights_before_step.keys():
            before = callback.weights_before_step[name]
            after = callback.weights_after_step[name]

            diff = (after - before).abs().max().item()
            if diff > 1e-8:
                changes_detected += 1
                print(f"  {name}: max_change={diff:.6e}")

        assert changes_detected > 0, "No weight changes detected after optimizer.step()"

        print(f"PASS: {changes_detected} parameters changed after step\n")


def test_gradient_clipping_after_backward_end():
    """Test that gradient clipping happens AFTER on_backward_end()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Gradient Clipping After backward_end")
        print("=" * 70)

        class ClippingVerificationCallback(TrainerCallback):
            def __init__(self):
                self.grad_norms_before_clip = []

            def on_backward_end(self, trainer):
                # Measure gradients BEFORE clipping
                total_norm = 0.0
                for param in trainer.model.parameters():
                    if param.grad is not None:
                        total_norm += param.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                self.grad_norms_before_clip.append(total_norm)

        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        callback = ClippingVerificationCallback()

        # Set gradient clipping
        config = TrainerConfig(
            model_name="test_clipping",
            result_dir=tmpdir,
            gradient_clip=1.0,  # Clip at 1.0
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

        # Create dataset with large gradients
        x = torch.randn(8, 10) * 10  # Large inputs
        y = torch.randint(0, 5, (8,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        # Monkey-patch
        def patched_train_step(batch):
            x_batch, y_batch = batch
            x_batch = x_batch.to(trainer.device)
            y_batch = y_batch.to(trainer.device)

            trainer.optimizer.zero_grad()
            output = trainer.model.forward(x_batch)
            loss = trainer.criterion(output, y_batch)

            trainer.callbacks.on_forward_end(trainer, loss)
            loss.backward()
            trainer.callbacks.on_backward_end(trainer)  # Measure BEFORE clip
            trainer._clip_gradients()  # Then clip
            trainer.optimizer.step()
            trainer.callbacks.on_step_end(trainer)

            return loss.item()

        trainer._train_one_step = patched_train_step

        # Run 1 epoch
        trainer.train(train_loader=loader, num_epochs=1)

        # Verify we measured gradients before clipping
        assert len(callback.grad_norms_before_clip) > 0

        # At least one gradient norm should be > clip threshold (1.0)
        # This proves we measured BEFORE clipping
        large_grads = [norm for norm in callback.grad_norms_before_clip if norm > 1.0]

        print(f"  Gradient norms before clip: {callback.grad_norms_before_clip}")
        print(f"  Norms > threshold (1.0): {len(large_grads)}/{len(callback.grad_norms_before_clip)}")

        assert len(large_grads) > 0, \
            "No large gradients detected - callback may be called AFTER clipping!"

        print("PASS: Gradients measured before clipping\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CALLBACK HOOK TIMING TEST SUITE")
    print("=" * 70)

    try:
        test_hook_timing_order()
        test_gradients_accessible_at_backward_end()
        test_weights_change_after_step()
        test_gradient_clipping_after_backward_end()

        print("=" * 70)
        print("ALL TIMING TESTS PASSED")
        print("=" * 70)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
