# Trainer Callback System

The trainer now uses a plugin-based callback system similar to TensorFlow/Keras callbacks. This allows extending trainer functionality without modifying core training code.

## Architecture

```
trainer_callback.py          - Base callback interface
trainer_early_stopping.py    - Early stopping plugin
trainer_gradient_monitor.py  - Gradient flow monitoring plugin
trainer.py                   - Core trainer with callback hooks
train_lm.py                  - CLI integration
```

## Available Callbacks

### 1. Early Stopping

Stops training when validation loss stops improving.

**CLI Usage:**
```bash
python train_lm.py \
  --early_stopping \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.001
```

**Programmatic Usage:**
```python
from trainer_early_stopping import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    patience=5,
    min_delta=0.001,
    restore_best=True
)

trainer = Trainer(..., callbacks=[callback])
```

### 2. Gradient Flow Monitoring

Monitors gradient flow through model layers, synced with snapshot saving.

**CLI Usage:**
```bash
python train_lm.py \
  --gradient_monitor \
  --snapshot_interval 1000 \
  --gradient_monitor_interval 500  # Optional: monitor every 500 steps
```

**Programmatic Usage:**
```python
from trainer_gradient_monitor import GradientMonitorCallback

callback = GradientMonitorCallback(
    monitor_at_snapshots=True,  # Monitor when snapshots are saved
    monitor_interval=500,        # Also monitor every 500 steps
    monitor_at_epochs=False
)

trainer = Trainer(..., callbacks=[callback])
```

## Creating Custom Callbacks

Create a new file `trainer_my_callback.py`:

```python
from trainer_callback import TrainerCallback

class MyCallback(TrainerCallback):
    def __init__(self, my_param):
        self.my_param = my_param

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss):
        print(f"Epoch {epoch} completed with loss {train_loss:.4f}")

    def on_backward_end(self, trainer):
        # Access gradients here (after backward, before clipping)
        pass

    def on_snapshot_save(self, trainer, epoch, step):
        # Return state to save with snapshot
        return {"my_state": self.my_param}

    def on_snapshot_load(self, trainer, checkpoint):
        # Restore state from checkpoint
        state = checkpoint.get("callback_state", {}).get("MyCallback", {})
        self.my_param = state.get("my_state", self.my_param)
```

## Callback Hook Points

Callbacks can hook into these training events:

```python
on_train_start(trainer)                                  # Once before training
on_train_end(trainer, result)                            # Once after training
on_epoch_start(trainer, epoch)                           # Before each epoch
on_epoch_end(trainer, epoch, train_loss, val_loss)      # After each epoch
on_batch_start(trainer, batch_idx)                      # Before each batch
on_batch_end(trainer, batch_idx, loss)                  # After each batch
on_forward_end(trainer, loss)                           # After forward pass
on_backward_end(trainer)                                 # After backward (gradients ready)
on_step_end(trainer)                                     # After optimizer.step()
on_snapshot_save(trainer, epoch, step) -> dict          # When saving snapshot
on_snapshot_load(trainer, checkpoint)                    # When loading snapshot
should_stop_training(trainer) -> bool                    # Check if should stop
```

## Complete Training Example

```bash
# Full-featured training with all callbacks
python train_lm.py \
  --dataset wikitext \
  --epochs 50 \
  --batch_size 32 \
  --snapshot_interval 1000 \
  --gradient_monitor \
  --early_stopping \
  --early_stop_patience 10
```

This will:
1. Train for up to 50 epochs
2. Save snapshots every 1000 steps
3. Monitor gradient flow at each snapshot
4. Stop early if no improvement for 10 epochs
5. Restore best weights when early stopping triggers

## Benefits

✅ **Modular** - Each feature is in its own file
✅ **Extensible** - Easy to add new features without modifying trainer.py
✅ **Composable** - Mix and match callbacks as needed
✅ **Synced** - Gradient monitoring syncs with snapshot saving
✅ **Persistent** - Callback state saved/restored with snapshots
✅ **Clean** - Trainer.py focuses on core training loop

## Migration Guide

### Old Code (Built-in Early Stopping)
```python
config = TrainerConfig(
    early_stop_patience=5,
    early_stop_min_delta=0.001,
    early_stop_restore_best=True
)
```

### New Code (Callback-based)
```python
from trainer_early_stopping import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    patience=5,
    min_delta=0.001,
    restore_best=True
)

trainer = Trainer(..., callbacks=[callback])
```

### Old Code (Built-in Gradient Monitor)
```python
config = TrainerConfig(
    enable_gradient_monitor=True
)
```

### New Code (Callback-based)
```python
from trainer_gradient_monitor import GradientMonitorCallback

callback = GradientMonitorCallback(
    monitor_at_snapshots=True
)

trainer = Trainer(..., callbacks=[callback])
```

## Files Changed

- `trainer_callback.py` - NEW: Base callback system
- `trainer_early_stopping.py` - NEW: Early stopping callback
- `trainer_gradient_monitor.py` - NEW: Gradient monitoring callback
- `trainer.py` - MODIFIED: Removed built-in early stopping, added callback hooks
- `train_lm.py` - MODIFIED: Added CLI options for callbacks
- `CALLBACKS.md` - NEW: This documentation
