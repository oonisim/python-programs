"""Regression tests for training behavior and data pipeline.

These tests lock in correctness for recent bug fixes. If any test fails,
it indicates that a regression has reintroduced a previously observed issue.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from model.constant import LABEL_IGNORE_VALUE  # noqa: E402
from model.lm import LanguageModel  # noqa: E402
from training.loader_translation import translation_collate_fn  # noqa: E402
from training.trainer import LanguageModelTrainer, TrainerConfig  # noqa: E402


class _DummyScheduler:
    """Minimal scheduler stub that counts how often it is stepped."""
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


def test_lm_scheduler_steps_per_batch_when_enabled():
    """Verify warmup schedules step once per batch.

    The warmup scheduler is defined in steps, not epochs. If the scheduler
    is only stepped per epoch, the learning rate can stay near zero for the
    entire run. This test asserts that the per-batch stepping path actually
    calls scheduler.step() once per training step.
    """
    # Build a tiny, CPU-only language model to keep the test fast.
    model = LanguageModel(
        vocab_size=32,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        max_seq_len=8,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = _DummyScheduler()
    config = TrainerConfig(
        model_name="test_lm_scheduler",
        step_scheduler_per_batch=True,
        snapshot_per_epoch=False,
        delete_snapshots_after_training=True,
    )
    trainer = LanguageModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.NLLLoss(ignore_index=LABEL_IGNORE_VALUE),
        config=config,
        device="cpu",
        scheduler=scheduler,
    )

    # Generate a small batch. Shapes: input (B, T), target (B, T).
    input_ids = torch.randint(0, 32, (2, 5))
    target_ids = torch.randint(0, 32, (2, 5))

    # One training step should advance the scheduler exactly once.
    trainer._train_one_step((input_ids, target_ids))
    assert scheduler.step_calls == 1


def test_translation_collate_does_not_poison_decoder_input():
    """Ensure padding masks do not corrupt decoder inputs.

    translation_collate_fn masks padded target positions with LABEL_IGNORE_VALUE
    for loss masking. If those masked values leak into decoder_input, the
    embedding lookup will receive invalid indices (e.g., -100), which is a
    hard runtime error. This test ensures decoder_input remains valid.
    """
    source_pad_id = 0
    target_pad_id = 0

    # Build a small batch with different target lengths to force padding.
    batch = [
        {"source_ids": torch.tensor([1, 2, 3]), "target_ids": torch.tensor([5, 6, 7])},
        {"source_ids": torch.tensor([4, 5]), "target_ids": torch.tensor([8])},
    ]

    # Collate and pad sequences.
    out = translation_collate_fn(
        batch=batch,
        source_pad_id=source_pad_id,
        target_pad_id=target_pad_id,
    )

    target_ids = out["target_ids"]
    decoder_input = target_ids[:, :-1]

    # Decoder inputs must never include LABEL_IGNORE_VALUE.
    assert torch.all(decoder_input != LABEL_IGNORE_VALUE)
