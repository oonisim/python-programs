# Overfitting Detection - Early Stopping Enhancement

## Overview

Enhanced the `EarlyStoppingCallback` to detect overfitting by monitoring when `val_loss - train_loss` keeps growing for N consecutive epochs. This provides early detection of model overfitting before performance severely degrades.

## How It Works

### Standard Early Stopping (Already Existed)
- Monitors: Validation loss not improving for N epochs
- Triggers: When `val_loss >= best_val_loss + min_delta` for `patience` epochs
- Use case: Model stopped learning or getting worse

### New: Overfitting Detection
- Monitors: Gap between validation and training loss
- Computes: `gap = val_loss - train_loss`
- Triggers: When `gap` increases by at least `overfit_min_delta` for `overfit_patience` consecutive epochs
- Use case: Model is memorizing training data instead of generalizing

### Why This Matters

**Typical Overfitting Pattern:**
```
Epoch 1: train=2.5, val=2.7, gap=0.2  ✓ Normal
Epoch 2: train=2.0, val=2.6, gap=0.6  ⚠️ Gap growing
Epoch 3: train=1.5, val=2.8, gap=1.3  ⚠️ Gap growing
Epoch 4: train=1.0, val=3.0, gap=2.0  🛑 Stop! Overfitting detected
```

Training loss keeps decreasing while validation loss increases or stagnates. This is the hallmark of overfitting.

## Usage

### Config Fields (TrainingConfig)

```python
enable_early_stopping: bool = False
early_stop_patience: int = 5
early_stop_min_delta: float = 0.001
early_stop_restore_best: bool = True
early_stop_overfit_patience: int = 0  # 0 = disabled (default)
early_stop_overfit_min_delta: float = 0.01
```

### CLI Arguments

```bash
# Standard early stopping
python train_lm.py --dataset wikitext \
    --early_stopping \
    --early_stop_patience 5 \
    --early_stop_min_delta 0.001

# With overfitting detection
python train_lm.py --dataset wikitext \
    --early_stopping \
    --early_stop_patience 5 \
    --early_stop_overfit_patience 3 \
    --early_stop_overfit_min_delta 0.05
```

### Direct API Usage

```python
from training.trainer_early_stopping import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    patience=5,                     # Stop if val loss doesn't improve for 5 epochs
    min_delta=0.001,               # Minimum improvement threshold
    restore_best=True,             # Restore best weights when stopping
    overfit_patience=3,            # Stop if gap grows for 3 epochs
    overfit_min_delta=0.05         # Minimum gap increase to count
)
```

## Parameters

### `overfit_patience` (int, default: 0)
- Number of consecutive epochs with increasing val-train gap before stopping
- Set to 0 to disable overfitting detection (default)
- Recommended: 3-5 epochs for most tasks
- Requires validation data to work

### `overfit_min_delta` (float, default: 0.01)
- Minimum increase in gap to count as overfitting trend
- Too small: False positives from noise
- Too large: Misses gradual overfitting
- Recommended: 0.01-0.1 depending on loss scale

## Examples

### Example 1: Conservative Detection (Default)
```bash
python train_lm.py --dataset wikitext \
    --early_stopping \
    --early_stop_patience 10
# Only stops when val loss stops improving
# Overfitting detection disabled by default
```

### Example 2: Aggressive Overfitting Detection
```bash
python train_lm.py --dataset wikitext \
    --early_stopping \
    --early_stop_patience 10 \
    --early_stop_overfit_patience 3 \
    --early_stop_overfit_min_delta 0.02
# Stops after 3 epochs of gap growing by 0.02+
```

### Example 3: Balanced Approach
```bash
python train_lm.py --dataset wikitext \
    --early_stopping \
    --early_stop_patience 7 \
    --early_stop_overfit_patience 5 \
    --early_stop_overfit_min_delta 0.05
# Standard: stop if no improvement for 7 epochs
# Overfitting: stop if gap grows for 5 epochs
```

## Output Examples

### When Overfitting Detected:
```
Epoch 5 | Step 1500/2000 | Train Loss: 0.8234 | Val Loss: 2.1543
  Early stopping: no improvement for 2/10 epochs (best: 2.0123 at epoch 3)
  Overfitting: gap increasing for 3/3 epochs (gap: 1.3309, increase: +0.2145)

Early stopping triggered at epoch 5
  Reason: Overfitting detected (val-train gap growing for 3 epochs)
  Current gap: 1.3309
  Train loss: 0.8234, Val loss: 2.1543
  Restored best weights from epoch 3 (loss: 2.0123)
```

### When Gap Stops Growing:
```
Epoch 6 | Step 1800/2000 | Train Loss: 0.7123 | Val Loss: 1.9876
  Early stopping: no improvement for 3/10 epochs (best: 2.0123 at epoch 3)
  Overfitting counter reset (gap decreased)
```

## Implementation Details

### State Tracking
```python
self.overfit_counter = 0        # Consecutive epochs with growing gap
self.prev_gap = None            # Previous epoch's gap (val - train)
```

### Logic Flow (on_epoch_end)
```python
# 1. Check standard early stopping (val loss not improving)
if current_loss < best_loss - min_delta:
    # Improvement: reset counter, save weights
else:
    counter += 1
    if counter >= patience:
        stop_training()

# 2. Check overfitting detection (gap growing)
if overfit_patience > 0 and val_loss is not None:
    current_gap = val_loss - train_loss
    if prev_gap is not None:
        gap_increase = current_gap - prev_gap
        if gap_increase > overfit_min_delta:
            overfit_counter += 1
            if overfit_counter >= overfit_patience:
                stop_training()
        else:
            overfit_counter = 0  # Reset
    prev_gap = current_gap
```

### Snapshot Persistence
Both standard and overfitting detection states are saved to snapshots:
```python
state = {
    'counter': self.counter,
    'best_loss': self.best_loss,
    'best_epoch': self.best_epoch,
    'overfit_counter': self.overfit_counter,  # New
    'prev_gap': self.prev_gap,                # New
    'best_weights': self.best_weights
}
```

## Testing

Run the test suite:
```bash
python3 test/test_overfit_detection.py
```

Tests cover:
1. Overfitting detection triggers after N epochs
2. Counter resets when gap stops growing
3. Detection disabled by default (overfit_patience=0)

## When To Use

### Enable Overfitting Detection When:
- ✅ Training on small datasets (high risk of overfitting)
- ✅ Using high-capacity models (many parameters)
- ✅ Training for many epochs
- ✅ Validation loss starts diverging from training loss
- ✅ You want early warning of memorization

### Keep Disabled (Default) When:
- ❌ Dataset is very large (overfitting unlikely)
- ❌ Model is underparameterized (can't overfit)
- ❌ Training for few epochs only
- ❌ Using strong regularization (dropout, weight decay)
- ❌ You want to explore model capacity limits

## Integration with Other Features

### Works With:
- ✅ Standard early stopping (both criteria checked independently)
- ✅ Weight monitoring (frozen weight detection)
- ✅ Gradient monitoring (gradient flow)
- ✅ Snapshot system (state persisted across restores)
- ✅ Best weight restoration (restores best epoch when triggered)

### Interaction with Standard Early Stopping:
- Both criteria are checked independently
- Whichever triggers first will stop training
- If overfitting detected first, message indicates the reason
- Best weights from lowest validation loss are always restored

## Files Modified

1. **training/trainer_early_stopping.py**
   - Added `overfit_patience` and `overfit_min_delta` parameters
   - Added `overfit_counter` and `prev_gap` state tracking
   - Enhanced `on_epoch_end()` with gap monitoring logic
   - Updated snapshot save/load for new state

2. **training/train_lm.py**
   - Added config fields: `early_stop_overfit_patience`, `early_stop_overfit_min_delta`
   - Added CLI arguments: `--early_stop_overfit_patience`, `--early_stop_overfit_min_delta`
   - Pass parameters to EarlyStoppingCallback constructor

3. **test/test_overfit_detection.py** (NEW)
   - Comprehensive test suite for overfitting detection
   - Tests trigger conditions, counter reset, and disabled state

## Future Enhancements

Potential improvements:
- [ ] Adaptive threshold based on loss scale
- [ ] Exponential moving average of gap (smoother detection)
- [ ] Warn N epochs before stopping (give user heads-up)
- [ ] Log gap to TensorBoard for visualization
- [ ] Multi-stage detection (warn -> stop)

## Summary

**What Changed:**
- Early stopping now has two independent stopping criteria
- New overfitting detection monitors val-train gap growth
- Disabled by default (set `overfit_patience > 0` to enable)
- Works seamlessly with existing early stopping logic

**Why It Matters:**
- Catches overfitting early before severe performance degradation
- Saves compute by stopping futile training
- Restores best weights automatically
- Provides clear console feedback on why training stopped

**How To Use:**
```bash
# Enable with recommended defaults
python train_lm.py --dataset wikitext \
    --early_stopping \
    --early_stop_overfit_patience 3
```
