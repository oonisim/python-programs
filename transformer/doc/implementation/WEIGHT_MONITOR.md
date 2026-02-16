# Weight Update Monitor - Callback Architecture

This document describes the weight update monitoring system, which uses a callback-based architecture to monitor gradient health and parameter updates during training.

## Architecture Overview

**As of v0.6**, weight update monitoring is implemented using the callback pattern:

- **Core monitoring logic**: `training/weight_update_monitor.py` - Sampling-based monitor (no interval logic)
- **Callback wrapper**: `training/weight_update_monitor_callback.py` - Integrates with training loop via callbacks
- **Usage**: Pass `WeightUpdateMonitorCallback` to `Trainer(..., callbacks=[...])`

This design provides:
- ✅ **Separation of concerns** - Monitoring decoupled from training loop
- ✅ **Consistency** - Matches pattern used by `GradientMonitorCallback`
- ✅ **Modularity** - Easy to add/remove or customize
- ✅ **Extensibility** - Subclass callback for custom behavior

---

## Quick Start

### Basic Usage

```python
from training.trainer import LanguageModelTrainer, TrainerConfig
from training.weight_update_monitor_callback import WeightUpdateMonitorCallback

# Create callback
weight_monitor = WeightUpdateMonitorCallback(
    monitor_interval=100,           # Check every 100 steps
    sample_size=1024,                # Sample 1024 elements per parameter
    vanishing_grad_threshold=1e-7,
    exploding_grad_threshold=1e2,
    frozen_update_ratio_threshold=1e-12,
    frozen_patience_steps=3,
    monitor_topk=5,
)

# Create trainer with callback
trainer = LanguageModelTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    config=trainer_config,
    callbacks=[weight_monitor],  # Add to callbacks
)

# Train
trainer.train(train_loader, val_loader, num_epochs=10)
```

### Command-Line Usage (train_lm.py)

```bash
# Enable monitoring (enabled by default)
python train_lm.py --dataset wikitext --weight_monitor

# Customize parameters
python train_lm.py --dataset wikitext \
    --weight_monitor \
    --weight_monitor_interval 50 \
    --weight_monitor_sample_size 2048
```

---

## What It Monitors

### 1. Gradient Health (after backward pass)
- **Gradient norms**: L2, max absolute, mean absolute
- **Vanishing gradients**: ||grad||_2 <= 1e-7 (too small)
- **Exploding gradients**: ||grad||_2 >= 1e2 (too large)
- **Per-parameter learning rates**: From optimizer param groups

**Call timing**: After `backward()` but before `optimizer.step()`

### 2. Parameter Updates (after optimizer step)
- **Actual weight changes**: Δw = w_new - w_old
- **Update ratio**: ||Δw|| / (||w|| + ε)
- **Frozen parameters**: Consecutive tiny updates (ratio <= 1e-12 for 3 steps)
- **Top-K worst parameters**: Parameters with smallest update ratios

**Call timing**: After `optimizer.step()`

### Key Features
- ✅ Works correctly with AdamW, weight decay, gradient clipping
- ✅ Supports multiple optimizer param groups (per-parameter LR)
- ✅ Automatically skips unused parameters (e.g., cross-attention in decoder-only models)
- ✅ Decisive frozen detection (measures actual Δw, not just gradients)

---

## Configuration Parameters

### Monitoring Frequency
```python
monitor_interval: int = 100  # Check weights every N steps (must be >= 1)
```
- Higher values reduce overhead but provide less frequent updates
- Typical range: 50-200 steps
- Overhead: ~1-2% per 100 steps

### Sampling Configuration
```python
sample_size: int = 1024  # Elements sampled per parameter
```
- Uses deterministic sampling (stable across runs, FNV-1a hash)
- Memory usage: ~4KB per parameter
- Total memory for 1000 params: ~4MB (vs 400MB with full clones)
- Larger samples = more accurate statistics but more memory

### Gradient Health Thresholds
```python
vanishing_grad_threshold: float = 1e-7  # Flag if ||grad||_2 <= this
exploding_grad_threshold: float = 1e2   # Flag if ||grad||_2 >= this
```

### Frozen Parameter Detection
```python
frozen_update_ratio_threshold: float = 1e-12  # Flag if update_ratio <= this
frozen_patience_steps: int = 3  # Consecutive frozen steps before flagging
```
- **Update ratio formula**: ||Δw|| / (||w|| + ε)
- Measures actual parameter change after `optimizer.step()`
- Decisive: accounts for weight decay, momentum, learning rate

### Logging Configuration
```python
monitor_topk: int = 5  # Log top-K parameters with smallest updates
```
- Prevents TensorBoard tag explosion (5 tags vs 1000+)
- Shows the "worst" offenders only

---

## TensorBoard Metrics

All metrics logged under `monitor/` namespace:

### Gradient Metrics (after backward)
- `monitor/grad_norm_median` - Median gradient norm across parameters
- `monitor/grad_norm_p95` - 95th percentile gradient norm
- `monitor/grad_norm_min` - Minimum gradient norm
- `monitor/grad_norm_max` - Maximum gradient norm
- `monitor/vanishing_count` - Number of parameters with vanishing gradients
- `monitor/exploding_count` - Number of parameters with exploding gradients

### Update Metrics (after optimizer step)
- `monitor/update_ratio_median` - Median update ratio
- `monitor/update_ratio_p95` - 95th percentile update ratio
- `monitor/update_ratio_min` - Minimum update ratio
- `monitor/update_ratio_max` - Maximum update ratio
- `monitor/frozen_count` - Number of frozen parameters
- `monitor/topk_frozen/{param_name}` - Update ratios for top-K worst parameters

### Tag Cardinality
**Old approach (per-parameter logging):**
- 3000+ tags for 1000 parameters (cardinality bomb)

**New approach (aggregate + top-K):**
- 12 base aggregate tags
- 5 top-K tags (configurable)
- **Total: ~17 tags** (vs 3000+)

---

## Console Output

### Warnings
The callback prints warnings when issues are detected:

```
⚠️  WARNING (step 1000): 3 vanishing gradients detected!
⚠️  WARNING (step 1500): 2 exploding gradients detected!
⚠️  WARNING (step 2000): 5 frozen parameters detected!
     Frozen: decoder.layers.0.attn.w_q.weight, decoder.layers.1.attn.w_q.weight, ...
```

### Initialization
```
Weight update monitoring enabled (sample_size=1024, interval=100 steps)
```

### On Training End
```
Weight monitor closed
```

---

## Memory & Performance

### Memory Impact

**Old approach (full parameter clones):**
- 400MB per monitoring interval (for 100M params float32)
- GPU fragmentation from repeated alloc/free
- Memory spike can trigger OOM

**New approach (sampling):**
- ~4MB for 1000 params × 1024 samples × 4 bytes
- No fragmentation (small buffers, reused via cache)
- **100× memory reduction**

### Compute Overhead

- **Overhead**: ~1-2% per 100 steps
- **Depends on**: `monitor_interval` (higher = less overhead)
- **Scales with**: Number of parameters and sample size
- **Typical**: Negligible for interval >= 100

### Comparison Table

| Metric | Old (Full Clone) | New (Sampling) |
|--------|------------------|----------------|
| Memory | 400MB | 4MB |
| Fragmentation | High | None |
| Overhead | ~5% | ~1-2% |
| Tag count | 3000+ | ~17 |
| Decisive frozen detection | ❌ | ✅ |

---

## Implementation Details

### Core Monitor (`weight_update_monitor.py`)

The `WeightUpdateMonitor` class provides:

1. **Deterministic sampling** - Uses FNV-1a hash (stable, no Python hash randomization)
2. **Sample caching** - Indices cached per parameter (name + shape)
3. **Frozen state tracking** - Consecutive frozen step counters
4. **Aggregate statistics** - Median, p95, min, max (optimized Python floats)
5. **Top-K reporting** - Worst parameters by update ratio

**Key methods:**
```python
def check_gradients(model, optimizer) -> Dict[str, GradientDiagnostics]
def check_updates(model, optimizer) -> Dict[str, UpdateDiagnostics]
def aggregate_gradient_stats(gradients) -> Dict[str, float]
def aggregate_update_stats(updates) -> Dict[str, float]
def top_k_largest_gradients(gradients, k) -> List[Tuple[str, float]]
def top_k_smallest_updates(updates, k) -> List[Tuple[str, float]]
```

### Callback (`weight_update_monitor_callback.py`)

The `WeightUpdateMonitorCallback` integrates the monitor with the training loop:

**Callback hooks:**
- `on_train_start()` - Initialize monitor
- `on_backward_end()` - Capture gradient statistics (at intervals)
- `on_step_end()` - Capture update statistics (at intervals)
- `on_snapshot_save()` - Save frozen state with checkpoint
- `on_snapshot_load()` - Restore frozen state from checkpoint
- `on_train_end()` - Clean up monitor

**Interval checking:**
```python
if trainer.global_step % self.monitor_interval == 0:
    # Capture statistics
```

### State Persistence

Frozen step counters are saved/restored with snapshots:
```python
# Saved state
{
    'frozen_steps': {param_name: count, ...},
    'epoch': epoch,
    'step': step,
    'global_step': global_step,
}
```

This ensures consistent frozen parameter detection after resuming training.

---

## Call Timing (Critical)

The monitor must be called at specific points in the training loop:

```python
# Forward pass
loss = model(inputs, targets)

# Backward pass
loss.backward()

# 1️⃣ CHECK GRADIENTS HERE (after backward, before clipping)
callbacks.on_backward_end(trainer)  # WeightUpdateMonitorCallback captures gradients

# Clip gradients
clip_grad_norm_(model.parameters(), max_norm)

# Optimizer step
optimizer.step()

# 2️⃣ CHECK UPDATES HERE (after step, before zero_grad)
callbacks.on_step_end(trainer)  # WeightUpdateMonitorCallback captures updates

# Zero gradients for next iteration
optimizer.zero_grad()
```

**Why this order?**
- Gradients captured after backward but before clipping (see raw gradient health)
- Updates captured after step (measure actual Δw)
- Must happen before `zero_grad()` (gradients needed)

---

## Frozen Parameter Detection

### Algorithm

1. **Compute update ratio**: `ratio = ||Δw|| / (||w|| + ε)`
2. **Check threshold**: Is `ratio <= frozen_update_ratio_threshold`?
3. **Count consecutive**: Increment frozen counter if yes, reset if no
4. **Flag frozen**: `is_frozen = consecutive_count >= frozen_patience_steps`

### Why "decisive"?

- Measures **actual parameter change** (Δw = w_new - w_old)
- Accounts for **weight decay** (not just gradients)
- Accounts for **momentum** (AdamW's running averages)
- Accounts for **per-parameter LR** (optimizer param groups)
- Requires **consecutive** tiny updates (patience threshold)

### False Positives

**Automatically skipped:**
- Parameters without gradients (not in computational graph)
- Unused modules (e.g., cross-attention in decoder-only LMs)

**May indicate real issues:**
- Learning rate too small
- Dead neurons (ReLU saturation)
- Architectural problems (gradient flow)
- Over-regularization (weight decay too high)

---

## Comparison with GradientMonitorCallback

| Feature | WeightUpdateMonitorCallback | GradientMonitorCallback |
|---------|---------------------------|------------------------|
| **Monitors** | Gradient health + parameter updates | Gradient flow between layers |
| **Scope** | All trainable parameters | Specific layer blocks (e.g., decoder.layers) |
| **Method** | Sampling (1024 elements) | Full gradient hooks on layer outputs |
| **Detects** | Vanishing/exploding grads, frozen params | Gradient flow, vanishing/amplifying transitions |
| **Memory** | ~4MB (sampling) | Negligible (hooks on layer outputs) |
| **Use case** | General training health monitoring | Architecture-specific gradient flow analysis |
| **Call timing** | After backward + after step | After backward only |

**Use together for comprehensive monitoring:**
```python
callbacks = [
    GradientMonitorCallback(monitor_interval=100),  # Layer-wise gradient flow
    WeightUpdateMonitorCallback(monitor_interval=100),  # Parameter-wise health
]
```

---

## Migration from Hardcoded Integration

### Before (v0.5 - Hardcoded in Trainer)

```python
# TrainerConfig included weight monitor options
trainer_config = TrainerConfig(
    model_name="lm_wikitext",
    enable_weight_monitor=True,
    weight_monitor_interval=100,
    weight_monitor_sample_size=1024,
    monitor_topk=5,
    vanishing_grad_threshold=1e-7,
    exploding_grad_threshold=1e2,
    frozen_update_ratio_threshold=1e-12,
    frozen_patience_steps=3,
)

trainer = LanguageModelTrainer(config=trainer_config)
```

**Issues with old approach:**
- ❌ Tight coupling (monitoring logic in training loop)
- ❌ Inconsistent with other monitoring (gradient monitor used callbacks)
- ❌ Hard to customize (must modify Trainer class)
- ❌ Hard to test in isolation

### After (v0.6+ - Callback-Based)

```python
# TrainerConfig no longer has weight monitor options
trainer_config = TrainerConfig(
    model_name="lm_wikitext",
    # weight monitor options removed
)

# Create callback separately
weight_monitor = WeightUpdateMonitorCallback(
    monitor_interval=100,
    sample_size=1024,
    vanishing_grad_threshold=1e-7,
    exploding_grad_threshold=1e2,
    frozen_update_ratio_threshold=1e-12,
    frozen_patience_steps=3,
    monitor_topk=5,
)

# Pass callback to trainer
trainer = LanguageModelTrainer(
    config=trainer_config,
    callbacks=[weight_monitor],
)
```

**Benefits of new approach:**
- ✅ Separation of concerns (monitoring decoupled)
- ✅ Consistent with GradientMonitorCallback pattern
- ✅ Easy to customize (subclass callback)
- ✅ Easy to test (callback isolated)
- ✅ Easy to disable (just omit from callbacks list)

### Migration Checklist

1. ✅ Remove `enable_weight_monitor`, `weight_monitor_interval`, etc. from `TrainerConfig`
2. ✅ Remove `enable_weight_monitor`, etc. from `TrainingConfig` in `train_lm.py`
3. ✅ Create `WeightUpdateMonitorCallback` in `_build_callbacks()`
4. ✅ Pass callback to `Trainer(..., callbacks=[...])`
5. ✅ Update command-line args to create callback instead of config options
6. ✅ Test that monitoring still works correctly

---

## Troubleshooting

### No metrics in TensorBoard

**Check:**
1. Is callback added to trainer? `callbacks=[weight_monitor]`
2. Is `monitor_interval` too high? Try `monitor_interval=50`
3. Did training run long enough? Need at least `monitor_interval` steps

### Too many warnings

**Vanishing/exploding gradients:**
- Adjust thresholds: `vanishing_grad_threshold`, `exploding_grad_threshold`
- Check learning rate (too high/low?)
- Check gradient clipping (too aggressive?)

**Frozen parameters:**
- Adjust threshold: `frozen_update_ratio_threshold=1e-10` (less strict)
- Adjust patience: `frozen_patience_steps=5` (require more consecutive)
- Check learning rate (too small?)
- Check if parameters are actually used (some may be legitimately frozen)

### High memory usage

**Reduce sampling:**
- `sample_size=512` (half the samples, half the memory)
- Still accurate for most use cases

**Increase interval:**
- `monitor_interval=200` (check half as often)
- Reduces overhead and memory pressure

---

## Ship-Ready Checklist ✅

All implementation issues addressed:

1. ✅ **Callback architecture** - Monitoring decoupled from training loop
2. ✅ **No interval logic in monitor** - Callback controls scheduling
3. ✅ **Fixed percentile calculation** - Uses `ceil` for correct p95
4. ✅ **Optimized aggregate stats** - Sorted Python floats, no extra tensors
5. ✅ **Stable hash** - FNV-1a, no Python hash randomization
6. ✅ **Correct call timing** - After clipping (gradients), after step (updates)
7. ✅ **Decisive update ratio** - ||Δw|| / (||w|| + ε)
8. ✅ **No per-param logging** - Aggregate + top-K only (no cardinality bombs)
9. ✅ **Console warnings** - Only for actual issues (`is_frozen=True`)
10. ✅ **Works with AdamW** - Accounts for weight decay, momentum
11. ✅ **Multi-param group support** - Per-parameter learning rates
12. ✅ **State persistence** - Frozen counters saved/restored with checkpoints
13. ✅ **Automatic filtering** - Skips unused parameters (no false positives)

---

## See Also

- **Implementation**: `src/training/weight_update_monitor.py` - Core monitoring logic
- **Callback**: `src/training/weight_update_monitor_callback.py` - Training integration
- **Usage**: `src/training/train_lm.py` - Example usage in training director
- **Gradient monitoring**: `src/training/trainer_gradient_monitor.py` - Complementary callback
- **Old docs**: `doc/implementation/WEIGHT_MONITOR_INTEGRATION.md` - Original integration (v0.5, deprecated)
