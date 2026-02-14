# Weight Update Monitor Integration Guide

This document shows the exact changes needed to integrate the ship-ready `WeightUpdateMonitor` into the trainer.

## Summary

- ✅ **File created**: `training/weight_update_monitor.py` (ship-ready, no interval logic)
- 📝 **File to modify**: `training/trainer.py` (5 sections)
- 📝 **File to modify**: `training/train_lm.py` (2 sections)

---

## Part 1: Modify `training/trainer.py`

### A) Add import (after line 51)

```python
# After: from training.trainer_callback import CallbackList, TrainerCallback
from training.weight_update_monitor import WeightUpdateMonitor
```

### B) Extend TrainerConfig (around line 82-90)

Add these fields to the `TrainerConfig` dataclass:

```python
@dataclass
class TrainerConfig:
    model_name: str = "transformer"
    result_dir: str = "result"
    gradient_clip: Optional[float] = 1.0
    log_interval: int = 100
    snapshot_interval: int = 0
    snapshot_per_epoch: bool = True
    keep_last_n_snapshots: int = 5
    delete_snapshots_after_training: bool = True
    max_steps: Optional[int] = None

    # ADD THESE LINES:
    enable_weight_monitor: bool = False
    weight_monitor_interval: int = 100
    weight_monitor_sample_size: int = 1024
    monitor_topk: int = 5
    vanishing_grad_threshold: float = 1e-7
    exploding_grad_threshold: float = 1e2
    frozen_update_ratio_threshold: float = 1e-12
    frozen_patience_steps: int = 3
```

### C) Initialize monitor in __init__ (after line 148, after writer creation)

```python
# After: self.writer = SummaryWriter(log_dir=self.model_root_dir / "runs")

# ADD THESE LINES:
self.weight_monitor: Optional[WeightUpdateMonitor] = None
if self.config.enable_weight_monitor:
    self.weight_monitor = WeightUpdateMonitor(
        sample_size=self.config.weight_monitor_sample_size,
        vanishing_grad_threshold=self.config.vanishing_grad_threshold,
        exploding_grad_threshold=self.config.exploding_grad_threshold,
        frozen_update_ratio_threshold=self.config.frozen_update_ratio_threshold,
        frozen_patience_steps=self.config.frozen_patience_steps,
    )
    print(f"Weight update monitoring enabled (sample_size={self.config.weight_monitor_sample_size}, "
          f"interval={self.config.weight_monitor_interval} steps)")
```

### D) Modify _train_one_step (around line 282-291)

```python
# EXISTING CODE:
loss.backward()
self.callbacks.on_backward_end(self)
self._clip_gradients()

# ADD AFTER _clip_gradients():
if self.weight_monitor and (self.global_step % self.config.weight_monitor_interval == 0):
    self._log_weight_monitor_gradients(self.global_step)

# EXISTING CODE:
self.optimizer.step()

# ADD AFTER optimizer.step():
if self.weight_monitor and (self.global_step % self.config.weight_monitor_interval == 0):
    self._log_weight_monitor_updates(self.global_step)

# EXISTING CODE:
self.callbacks.on_step_end(self)
```

### E) Add monitoring methods (insert BEFORE _log_epoch_summary, around line 415)

```python
# ========================================================================
# Weight Monitoring Methods (aggregate only, no per-parameter tags)
# ========================================================================

def _log_weight_monitor_gradients(self, step: int) -> None:
    """Log aggregate gradient diagnostics (no per-param tags)."""
    assert self.weight_monitor is not None
    grad_diag = self.weight_monitor.check_gradients(self.model, self.optimizer)
    stats = WeightUpdateMonitor.aggregate_gradient_stats(grad_diag)

    if stats.get("count", 0.0) == 0.0:
        return

    self.writer.add_scalar("monitor/grad_median", stats["median"], step)
    self.writer.add_scalar("monitor/grad_p95", stats["p95"], step)
    self.writer.add_scalar("monitor/grad_min", stats["min"], step)
    self.writer.add_scalar("monitor/grad_max", stats["max"], step)
    self.writer.add_scalar("monitor/vanishing_count", stats["vanishing_count"], step)
    self.writer.add_scalar("monitor/exploding_count", stats["exploding_count"], step)

    # Optional: top-K per param (adds K tags)
    for name, val in WeightUpdateMonitor.top_k_largest_gradients(
        grad_diag, k=self.config.monitor_topk
    ):
        self.writer.add_scalar(f"monitor/topk_grad/{name}", val, step)


def _log_weight_monitor_updates(self, step: int) -> None:
    """Log aggregate update diagnostics (decisive frozen detection)."""
    assert self.weight_monitor is not None
    upd_diag = self.weight_monitor.check_updates(self.model, self.optimizer)
    stats = WeightUpdateMonitor.aggregate_update_stats(upd_diag)

    if stats.get("count", 0.0) == 0.0:
        return

    self.writer.add_scalar("monitor/update_ratio_median", stats["median"], step)
    self.writer.add_scalar("monitor/update_ratio_p95", stats["p95"], step)
    self.writer.add_scalar("monitor/update_ratio_min", stats["min"], step)
    self.writer.add_scalar("monitor/update_ratio_max", stats["max"], step)
    self.writer.add_scalar("monitor/frozen_count", stats["frozen_count"], step)

    # Optional: top-K per param (adds K tags)
    for name, val in WeightUpdateMonitor.top_k_smallest_updates(
        upd_diag, k=self.config.monitor_topk
    ):
        self.writer.add_scalar(f"monitor/topk_frozen/{name}", val, step)

    # Console warnings for frozen weights (only if detected)
    if stats["frozen_count"] > 0:
        frozen_params = [
            (name, diag.update_ratio)
            for name, diag in upd_diag.items()
            if diag.is_frozen
        ]
        frozen_params.sort(key=lambda x: x[1])

        print(f"  ⚠️  Step {step}: {stats['frozen_count']:.0f} frozen params "
              f"(update_ratio <= {self.config.frozen_update_ratio_threshold:.0e} "
              f"for {self.config.frozen_patience_steps} consecutive steps)")

        for name, ratio in frozen_params[:self.config.monitor_topk]:
            print(f"      {name}: {ratio:.2e}")
```

### F) Modify _log_epoch_summary (around line 434-436)

REMOVE these three lines:
```python
# DELETE: self._log_weight_stats(epoch)
# DELETE: self._log_gradient_flow(epoch)
# DELETE: self._log_weight_updates(epoch)
```

### G) Delete old methods (around line 438-548)

DELETE these entire methods:
- `def _log_weight_stats(self, epoch: int) -> None:` (lines 438-463)
- `def _log_gradient_flow(self, epoch: int) -> None:` (lines 465-510)
- `def _log_weight_updates(self, epoch: int) -> None:` (lines 512-548)

---

## Part 2: Modify `training/train_lm.py`

### A) Extend TrainingConfig (around line 105)

Add these fields to the `TrainingConfig` dataclass:

```python
@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    log_interval: int = 100
    snapshot_interval: int = 0
    keep_last_n_snapshots: int = 3
    delete_snapshots_after_training: bool = True
    # Callback options...
    enable_gradient_monitor: bool = False
    gradient_monitor_interval: int = 0
    enable_early_stopping: bool = False
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.001
    early_stop_restore_best: bool = True

    # ADD THESE LINES:
    enable_weight_monitor: bool = False
    weight_monitor_interval: int = 100
    weight_monitor_sample_size: int = 1024
    monitor_topk: int = 5
    vanishing_grad_threshold: float = 1e-7
    exploding_grad_threshold: float = 1e2
    frozen_update_ratio_threshold: float = 1e-12
    frozen_patience_steps: int = 3
```

### B) Add CLI arguments (in train_group, around line 675 after early_stop args)

```python
# ADD THESE ARGUMENTS:
train_group.add_argument(
    "--weight_monitor", action="store_true",
    help=(
        "Enable weight update monitoring to detect frozen weights. "
        "Uses a few MB of GPU memory (sample_size × num_params). "
        "Monitors actual Δw after optimizer.step() (works correctly "
        "with AdamW, weight decay, and gradient clipping)."
    )
)
train_group.add_argument(
    "--weight_monitor_interval", type=int, default=100,
    metavar="N",
    help="Monitor weight updates every N steps (must be >= 1). Default: 100"
)
train_group.add_argument(
    "--weight_monitor_sample_size", type=int, default=1024,
    metavar="SIZE",
    help="Number of sampled elements per parameter. Default: 1024"
)
```

### C) Pass config to TrainingConfig (in main(), around line 800)

```python
training_config = TrainingConfig(
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.lr,
    snapshot_interval=args.snapshot_interval,
    keep_last_n_snapshots=args.keep_last_n_snapshots,
    delete_snapshots_after_training=args.delete_snapshots_after_training,
    enable_gradient_monitor=args.gradient_monitor,
    gradient_monitor_interval=args.gradient_monitor_interval,
    enable_early_stopping=args.early_stopping,
    early_stop_patience=args.early_stop_patience,
    early_stop_min_delta=args.early_stop_min_delta,
    early_stop_restore_best=args.early_stop_restore_best,
    # ADD THESE LINES:
    enable_weight_monitor=args.weight_monitor,
    weight_monitor_interval=args.weight_monitor_interval,
    weight_monitor_sample_size=args.weight_monitor_sample_size,
)
```

### D) Pass to TrainerConfig in director (around line 375)

```python
trainer_config = TrainerConfig(
    model_name=f"lm_{self.dataset_key}",
    gradient_clip=self.training_config.gradient_clip,
    log_interval=self.training_config.log_interval,
    snapshot_interval=self.training_config.snapshot_interval,
    snapshot_per_epoch=True,
    keep_last_n_snapshots=self.training_config.keep_last_n_snapshots,
    delete_snapshots_after_training=self.training_config.delete_snapshots_after_training,
    # ADD THESE LINES:
    enable_weight_monitor=self.training_config.enable_weight_monitor,
    weight_monitor_interval=self.training_config.weight_monitor_interval,
    weight_monitor_sample_size=self.training_config.weight_monitor_sample_size,
    monitor_topk=self.training_config.monitor_topk,
    vanishing_grad_threshold=self.training_config.vanishing_grad_threshold,
    exploding_grad_threshold=self.training_config.exploding_grad_threshold,
    frozen_update_ratio_threshold=self.training_config.frozen_update_ratio_threshold,
    frozen_patience_steps=self.training_config.frozen_patience_steps,
)
```

---

## Usage

```bash
# Train WITHOUT monitoring (default, no overhead)
python train_lm.py --dataset wikitext --epochs 20

# Train WITH monitoring (decisive frozen weight detection)
python train_lm.py --dataset wikitext --epochs 20 --weight_monitor

# Custom interval and sample size
python train_lm.py --dataset wikitext --weight_monitor \
    --weight_monitor_interval 50 \
    --weight_monitor_sample_size 2048
```

---

## TensorBoard Tags

**Old (per-parameter, cardinality bomb):**
- ~3000 tags for 1000 parameters

**New (aggregate only):**
- 12 base tags (grad_median, grad_p95, etc.)
- Optional: 2×topk tags (topk_grad/{name}, topk_frozen/{name})
- **Total: 12-22 tags** (vs 3000+)

---

## Memory Impact

**Old `_log_weight_updates()`:**
- 400MB cloned every epoch (for 100M params float32)
- GPU fragmentation from repeated alloc/free

**New `WeightUpdateMonitor`:**
- ~4MB for 1000 params × 1024 samples × 4 bytes
- No fragmentation (small buffers, reused)
- **100× memory reduction**

---

## Ship-Ready ✅

All issues addressed:
1. ✅ Removed interval logic from monitor (trainer controls scheduling)
2. ✅ Fixed percentile calculation (uses ceil)
3. ✅ Optimized aggregate stats (sorted Python floats, no extra tensors)
4. ✅ Stable hash (FNV-1a, no Python hash randomization)
5. ✅ Correct call timing (after clipping, after step)
6. ✅ Decisive update_ratio formula: ||Δw||/(||w||+ε)
7. ✅ Deleted all per-param logging (cardinality bombs)
8. ✅ Console warnings only for actual issues (is_frozen=True)
9. ✅ Works correctly with AdamW, weight decay, gradient clipping
10. ✅ Supports multiple param groups (per-param LR)
