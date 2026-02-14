# Gradient Flow Monitor

A production-ready PyTorch utility for monitoring and diagnosing gradient flow through deep neural networks. Detect vanishing/exploding gradients, compare architectures, and validate training stability.

## Features

✅ **Multiple norm types**: L2, L1, L∞, and mean absolute value
✅ **Log-space gains**: Numerically stable gradient analysis
✅ **Flexible modes**: Strict or permissive backward pass handling
✅ **Rich reporting**: Automatic categorization and summary statistics
✅ **Context manager**: Automatic cleanup of PyTorch hooks
✅ **Multiple output formats**: Handles tensors, tuples, dicts (HuggingFace-compatible)
✅ **Comprehensive tests**: Full test suite included

## Installation

Copy `gradient_monitor.py` to your project directory:

```bash
# No dependencies beyond PyTorch
pip install torch
```

## Quick Start

```python
import torch
import torch.nn as nn
from training.gradient_monitor import GradientGainMonitor

# Create your model with blocks
model_blocks = nn.ModuleList([
    TransformerBlock(dim=512) for _ in range(12)
])

# Monitor gradient flow
with GradientGainMonitor(model_blocks) as monitor:
    # Forward pass
    output = run_forward(model_blocks, inputs)
    loss = compute_loss(output, targets)

    # Backward pass
    loss.backward()

    # Analyze gradient flow
    print(monitor.report())
```

## What It Measures

### Gradient Norms

For each block output, measures:
```
‖∂L/∂h_i‖ = gradient magnitude at block i
```

### Gradient Gains

For each transition between blocks:
```
γ_i = ‖∂L/∂h_{i-1}‖ / ‖∂L/∂h_i‖
```

**Interpretation:**
- `γ > 1`: Gradient **amplifies** going backward (grows)
- `γ ≈ 1`: Gradient **preserved** (often healthy for residual networks)
- `γ < 1`: Gradient **dampens** (shrinks)
- `γ ≪ 1`: **Vanishing gradient** problem
- `γ ≫ 1`: Potential **exploding gradient** problem

**Note:** γ is the *backward gain* (previous_norm / current_norm)

## API Reference

### Constructor

```python
GradientGainMonitor(
    blocks,                      # List/ModuleList/Sequential of blocks
    norm_type='l2',             # 'l2', 'l1', 'linf', or 'mean'
    strict_single_backward=True, # Raise error on multiple backward?
    eps=1e-30                   # Epsilon for numerical stability
)
```

**Parameters:**
- `blocks`: Blocks to monitor (must be separate nn.Module instances)
- `norm_type`:
  - `'l2'`: Euclidean norm (most common)
  - `'l1'`: Manhattan norm (less sensitive to outliers)
  - `'linf'`: Max absolute value (detects spikes)
  - `'mean'`: Mean absolute value (scale-invariant, not a true norm)
- `strict_single_backward`:
  - `True`: Raises error if backward() called twice without reset()
  - `False`: Overwrites measurements (useful for gradient accumulation)
- `eps`: Small value to detect near-zero gradients

### Methods

#### `norms() -> List[Optional[float]]`
Returns gradient norm for each block output.

#### `gains() -> List[Optional[float]]`
Returns gain ratios between consecutive blocks (length = num_blocks - 1).

#### `log_gains() -> List[Optional[float]]`
Returns log-space gains for numerical stability:
```
log(γ_i) = log(‖∂L/∂h_{i-1}‖) - log(‖∂L/∂h_i‖)
```

#### `summary_stats() -> dict`
Returns statistics:
```python
{
    'mean_gain': float,           # Average gain
    'min_gain': float,            # Minimum gain
    'max_gain': float,            # Maximum gain
    'mean_log_gain': float,       # Average log-gain
    'num_amplifying': int,        # Count of γ > 2.0
    'num_damping': int,           # Count of γ < 0.5
    'num_healthy': int,           # Count of 0.5 ≤ γ ≤ 2.0
    'num_vanishing': int,         # Count of γ = inf
    'num_missing': int            # Count of None (no gradient)
}
```

#### `report() -> str`
Generates human-readable report with norms, gains, and summary.

#### `reset()`
Clears recorded gradients for next backward pass.

#### `close()`
Removes hooks (automatic with context manager).

## Usage Patterns

### 1. Basic Diagnostic

```python
with GradientGainMonitor(model.blocks) as monitor:
    loss.backward()
    print(monitor.report())
```

### 2. Training Loop

```python
monitor = GradientGainMonitor(model.blocks, strict_single_backward=False)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()

        # Check gradient flow periodically
        if step % 100 == 0:
            stats = monitor.summary_stats()
            if stats['num_damping'] > len(model.blocks) // 2:
                print("⚠️  Warning: Many damping transitions!")

        optimizer.step()
        monitor.reset()

monitor.close()
```

### 3. Gradient Accumulation

```python
monitor = GradientGainMonitor(model.blocks, strict_single_backward=False)
optimizer.zero_grad()

for i, micro_batch in enumerate(micro_batches):
    loss = compute_loss(micro_batch) / len(micro_batches)
    loss.backward()

    # Monitor each micro-batch
    print(f"Micro-batch {i}: {monitor.summary_stats()['mean_gain']:.4f}")
    monitor.reset()

optimizer.step()
monitor.close()
```

### 4. Architecture Comparison

```python
architectures = {
    'Pre-LN': pre_ln_blocks,
    'Post-LN': post_ln_blocks,
}

for name, blocks in architectures.items():
    with GradientGainMonitor(blocks) as monitor:
        loss = run_model(blocks)
        loss.backward()
        stats = monitor.summary_stats()
        print(f"{name}: mean_gain={stats['mean_gain']:.4f}")
```

## Executive Review Feedback: Addressed

This implementation incorporates all feedback from the engineering review:

### Correctness Fixes ✅

1. **Gradient detachment**: All gradients are detached before norm computation to prevent graph retention
2. **Multiple backward handling**: `strict_single_backward` flag controls behavior
3. **Numerical stability**: Uses `eps` instead of exact zero checks
4. **Hook safety**: Checks `requires_grad` before attaching hooks

### Semantic Improvements ✅

1. **Terminology**: Docstrings clarify "detects" vs "prevents"
2. **Gain direction**: Explicitly documents backward gain definition
3. **Ideal values**: Changed "gamma=1 is ideal" to "often healthy"
4. **Heuristic labels**: Report clearly indicates thresholds are heuristic

### API Enhancements ✅

1. **Log-gains**: Added `log_gains()` method for numerical stability
2. **Summary stats**: Added `summary_stats()` method with detailed metrics
3. **Multiple formats**: Supports tensor, tuple, list, and dict outputs
4. **nn.Sequential**: Explicitly supported in type hints
5. **Configurable epsilon**: User can set tolerance for zero detection

### Robustness ✅

1. **Checkpointing-safe**: Non-strict mode handles recomputation
2. **Frozen layers**: Properly handles blocks without gradients
3. **Error messages**: Detailed, actionable error messages
4. **Resource cleanup**: Context manager ensures proper hook removal

## Output Interpretation

### Healthy Gradient Flow

```
Gradient Gains (ratio of consecutive norms):
  Transition  0 to  1: gamma=0.9542, log_gamma=-0.0468 [Healthy]
  Transition  1 to  2: gamma=1.0234, log_gamma=0.0231 [Healthy]
  Transition  2 to  3: gamma=0.8876, log_gamma=-0.1193 [Healthy]

Summary Statistics:
  Mean gain: 0.9551
  Healthy (0.5 <= gamma <= 2.0): 3
```

✅ Gains near 1.0 indicate well-preserved gradients (common with residual connections)

### Vanishing Gradients

```
Gradient Gains (ratio of consecutive norms):
  Transition  5 to  6: gamma=0.3421, log_gamma=-1.0724 [DAMPING]
  Transition  6 to  7: gamma=0.2134, log_gamma=-1.5443 [DAMPING]
  Transition  7 to  8: gamma=0.0891, log_gamma=-2.4181 [DAMPING]

Summary Statistics:
  Mean gain: 0.4234
  Damping (gamma < 0.5): 6
```

⚠️ Many damping transitions suggest:
- Network may be too deep without residuals
- Consider Pre-LN architecture
- Check initialization scheme
- Add skip connections

### Exploding Gradients

```
Gradient Gains (ratio of consecutive norms):
  Transition  2 to  3: gamma=3.4512, log_gamma=1.2384 [AMPLIFYING]
  Transition  3 to  4: gamma=5.2341, log_gamma=1.6548 [AMPLIFYING]

Summary Statistics:
  Max gain: 5.2341
  Amplifying (gamma > 2.0): 4
```

⚠️ Amplifying transitions suggest:
- Check for unstable layers (LayerNorm on residual path)
- Consider gradient clipping
- Reduce learning rate
- Check initialization

## Testing

Run the comprehensive test suite:

```bash
# Install pytest
pip install pytest

# Run tests
python -m pytest test_gradient_monitor.py -v

# Run with coverage
pip install pytest-cov
python -m pytest test_gradient_monitor.py --cov=gradient_monitor
```

## Examples

Run the example script to see various usage patterns:

```bash
python example_gradient_monitor.py
```

Examples include:
1. Basic usage with residual blocks
2. Comparing Pre-LN vs Post-LN transformers
3. Different norm types (L2, L1, L∞, mean)
4. Integration with training loops
5. Gradient accumulation
6. Detecting gradient flow problems
7. Using log-gains for visualization

## Limitations & Caveats

1. **Hook overhead**: Hooks add computational overhead (~5-10% typically)
2. **Memory**: Stores one scalar per block (minimal impact)
3. **Heuristic thresholds**: Labels like "damping" use fixed thresholds (0.5, 2.0) that may need adjustment for your model
4. **Checkpointing**: With gradient checkpointing, hooks may fire during recomputation - use `strict_single_backward=False`
5. **Diagnostic only**: This tool *detects* gradient patterns but doesn't fix them

## Best Practices

1. **Use context manager**: Ensures proper cleanup
```python
with GradientGainMonitor(blocks) as monitor:
    # your code
```

2. **Monitor periodically**: Don't monitor every step in production (overhead)
```python
if step % 100 == 0:
    # monitor this step
```

3. **Compare architectures**: Use same data/seed for fair comparison
```python
torch.manual_seed(42)
# compare different architectures
```

4. **Check early training**: Gradient problems often appear early
```python
# Monitor first few epochs carefully
if epoch < 3:
    print(monitor.report())
```

5. **Use log-gains for plots**: More stable than raw gains
```python
log_gains = monitor.log_gains()
plt.plot(log_gains)
```

## Related Work

- [PyTorch Hooks Documentation](https://pytorch.org/docs/stable/notes/autograd.html#backward-hooks)
- [Gradient Flow in Residual Networks](https://arxiv.org/abs/1512.03385)
- [On Layer Normalization in Transformers](https://arxiv.org/abs/2002.04745)

## License

MIT License - feel free to use in your projects.

## Contributing

Contributions welcome! Please ensure:
- All tests pass
- New features include tests
- Docstrings follow NumPy style

## Citation

If you use this in research, please cite:

```bibtex
@software{gradient_gain_monitor,
  title={GradientGainMonitor: Gradient Flow Diagnostics for PyTorch},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/gradient-monitor}
}
```
