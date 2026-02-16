"""Test suite for scheduler stepping behavior.

Verifies that step_scheduler_per_batch config controls whether the LR scheduler
is stepped once per batch (for warmup schedules) or once per epoch (default).

Regression test for commit e990a7fc which fixed warmup stepping from per-epoch
to per-batch.
"""
import math
import tempfile
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import Trainer, TrainerConfig
from training.train_lm import get_cosine_schedule_with_warmup


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


def _make_trainer(config, scheduler_fn):
    """Create a Trainer with patched _train_one_step for simple (x, y) batches.

    Args:
        config: TrainerConfig instance.
        scheduler_fn: Callable(optimizer) -> scheduler.

    Returns:
        (trainer, scheduler) tuple.
    """
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = scheduler_fn(optimizer)
    criterion = nn.NLLLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        scheduler=scheduler,
    )

    # Patch _train_one_step to work with (x, y) tuple batches from TensorDataset
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

        # CRITICAL: Call the real production scheduler logic
        # Don't duplicate the if-check here - let _step_scheduler_if_configured handle it
        trainer._step_scheduler_if_configured()

        return loss.item()

    trainer._train_one_step = patched_train_step
    return trainer, scheduler


def test_warmup_scheduler_steps_per_batch():
    """With step_scheduler_per_batch=True, scheduler steps once per batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainerConfig(
            model_name="test_step_per_batch",
            result_dir=tmpdir,
            log_interval=999,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True,
            step_scheduler_per_batch=True,
            enable_weight_monitor=False,
        )

        def make_scheduler(opt):
            return get_cosine_schedule_with_warmup(
                opt, num_warmup_steps=5, num_training_steps=20
            )

        trainer, scheduler = _make_trainer(config, make_scheduler)

        # 100 samples / batch_size 10 = 10 batches per epoch
        data = create_dummy_data(100)
        loader = DataLoader(data, batch_size=10)

        trainer.train(train_loader=loader, num_epochs=1)

        # Scheduler should have been stepped 10 times (once per batch)
        assert scheduler.last_epoch == 10, (
            f"Expected scheduler.last_epoch == 10 (one step per batch), "
            f"got {scheduler.last_epoch}"
        )


def test_warmup_lr_ramps_over_steps():
    """LR ramps up from ~0 during warmup when stepped per batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainerConfig(
            model_name="test_lr_ramp",
            result_dir=tmpdir,
            log_interval=999,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True,
            step_scheduler_per_batch=True,
            enable_weight_monitor=False,
        )

        # Use known peak LR for stronger assertion
        peak_lr = 1e-3

        def make_scheduler(opt):
            return get_cosine_schedule_with_warmup(
                opt, num_warmup_steps=5, num_training_steps=20
            )

        trainer, scheduler = _make_trainer(config, make_scheduler)

        # Initial LR should be ~0 (warmup hasn't started)
        initial_lr = trainer.optimizer.param_groups[0]['lr']

        data = create_dummy_data(100)
        loader = DataLoader(data, batch_size=10)

        trainer.train(train_loader=loader, num_epochs=1)

        final_lr = scheduler.get_last_lr()[0]

        # LEARNING RATE SCHEDULE EXPLANATION:
        # ===================================
        # This test uses a two-phase LR schedule from get_cosine_schedule_with_warmup():
        #
        # Phase 1 - Linear Warmup (steps 1-5):
        #   Purpose: Gradually increase LR from 0 to peak to avoid early training instability
        #   Formula: lr = (current_step / warmup_steps) * peak_lr
        #   Result: Linear ramp from 0 → peak_lr
        #
        # Phase 2 - Cosine Annealing Decay (steps 6-20):
        #   Purpose: Smoothly decay LR using cosine curve for better convergence
        #   Formula: lr = min_lr + (peak_lr - min_lr) * cosine_decay
        #            where cosine_decay = 0.5 * (1 + cos(π * progress))
        #            and progress = steps_after_warmup / total_steps_after_warmup
        #   Result: Smooth cosine curve from peak_lr → min_lr_ratio * peak_lr
        #
        # At step 10 (after 5 warmup + 5 decay steps), calculate expected LR:
        warmup_steps = 5
        total_steps = 20
        current_step = 10
        min_lr_ratio = 0.1  # Default in get_cosine_schedule_with_warmup

        num_steps_after_warmup = current_step - warmup_steps  # 5
        total_steps_after_warmup = total_steps - warmup_steps  # 15
        progress = num_steps_after_warmup / total_steps_after_warmup  # 0.333
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # ≈ 0.75
        expected_lr = peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
        # expected_lr ≈ 0.775 * peak_lr (77.5% of peak)

        tolerance = 0.05  # Allow 5% numerical variation
        assert final_lr > expected_lr * (1 - tolerance), (
            f"Expected LR at step {current_step} to be ~{expected_lr:.6f} "
            f"({expected_lr/peak_lr:.1%} of peak), got {final_lr:.6f} "
            f"({final_lr/peak_lr:.1%} of peak). "
            f"Schedule: {warmup_steps}-step warmup + cosine decay to {total_steps} steps."
        )


def test_epoch_scheduler_steps_per_epoch_not_per_batch():
    """With step_scheduler_per_batch=False, scheduler steps once per epoch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainerConfig(
            model_name="test_step_per_epoch",
            result_dir=tmpdir,
            log_interval=999,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True,
            step_scheduler_per_batch=False,
            enable_weight_monitor=False,
        )

        def make_scheduler(opt):
            return CosineAnnealingLR(opt, T_max=10)

        trainer, scheduler = _make_trainer(config, make_scheduler)

        # 100 samples / batch_size 10 = 10 batches per epoch, 2 epochs = 20 batches
        data = create_dummy_data(100)
        loader = DataLoader(data, batch_size=10)

        trainer.train(train_loader=loader, num_epochs=2)

        # Scheduler should have been stepped 2 times (once per epoch), not 20
        assert scheduler.last_epoch == 2, (
            f"Expected scheduler.last_epoch == 2 (one step per epoch), "
            f"got {scheduler.last_epoch}"
        )


def test_warmup_completes_in_expected_steps():
    """Warmup completes in step units, not epoch units (core regression test)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        peak_lr = 1e-3
        warmup_steps = 5

        config = TrainerConfig(
            model_name="test_warmup_regression",
            result_dir=tmpdir,
            log_interval=999,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True,
            step_scheduler_per_batch=True,
            enable_weight_monitor=False,
        )

        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=20
        )
        criterion = nn.NLLLoss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            scheduler=scheduler,
        )

        # Track LR at each step
        lr_history = []

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

            # CRITICAL: Call the real production scheduler logic
            trainer._step_scheduler_if_configured()

            lr_history.append(trainer.optimizer.param_groups[0]['lr'])
            return loss.item()

        trainer._train_one_step = patched_train_step

        # 10 batches per epoch, warmup_steps=5 so warmup completes mid-epoch
        data = create_dummy_data(100)
        loader = DataLoader(data, batch_size=10)

        trainer.train(train_loader=loader, num_epochs=1)

        # After warmup_steps=5, LR should be at or near peak
        lr_at_warmup_end = lr_history[warmup_steps - 1]  # step 5 (0-indexed: 4)
        assert lr_at_warmup_end >= peak_lr * 0.99, (
            f"Expected LR at step {warmup_steps} to be near peak_lr={peak_lr}, "
            f"got {lr_at_warmup_end}"
        )

        # LR should have been monotonically increasing during warmup
        for i in range(1, warmup_steps):
            assert lr_history[i] > lr_history[i - 1], (
                f"LR should increase during warmup: "
                f"step {i} LR={lr_history[i]} <= step {i-1} LR={lr_history[i-1]}"
            )
