# Language Model Training Execution Guide

This guide explains how to train the GPT-style decoder-only language model using the `run_train_lm.sh` wrapper script and the underlying `train_lm.py` program.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Using run_train_lm.sh](#using-run_train_lmsh)
3. [Viewing Available Options](#viewing-available-options)
4. [Option Reference](#option-reference)
5. [Model Presets](#model-presets)
6. [Advanced Usage](#advanced-usage)
7. [Monitoring Training](#monitoring-training)

## Quick Start

### Basic Training with Defaults

```bash
# From the transformer directory
cd /home/masa/repository/python-programs/transformer

# Train with small preset on WikiText-103
./run/run_train_lm.sh --preset small
```

### Quick Test Run

```bash
# Train tiny model on smaller WikiText-2 dataset
./run/run_train_lm.sh --preset tiny --dataset wikitext
```

## Using run_train_lm.sh

The `run_train_lm.sh` script is a convenience wrapper that:
- Sets up the Python environment (PYTHONPATH)
- Provides sensible defaults
- Runs training in the background with nohup
- Creates timestamped log files
- Enables monitoring features automatically

### Basic Syntax

```bash
./run/run_train_lm.sh [OPTIONS]
```

### Script Location

The script is located at: `/home/masa/repository/python-programs/transformer/run/run_train_lm.sh`

Always run it from the transformer project root directory.

## Viewing Available Options

### Show Help from Shell Script

```bash
./run/run_train_lm.sh --help
```

This displays the simplified options available through the shell wrapper.

### Show Detailed Help from Python Script

```bash
# Show all command-line arguments with descriptions
python src/training/train_lm.py --help

# Show comprehensive documentation about language models
python src/training/train_lm.py --info

# Show quick explanation with examples
python src/training/train_lm.py --explain
```

## Option Reference

### run_train_lm.sh Options

The shell script accepts the following command-line options:

#### --preset PRESET
**Objective:** Select a predefined model architecture configuration.

**Available Values:**
- `tiny` - ~16M parameters (d_model=256, num_heads=4, num_layers=4, d_ff=1024, max_seq_len=256)
  - Suitable for WikiText-2 and quick experimentation
- `small` - ~45-50M parameters (d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=512)
  - Option A: Balanced model for WikiText-103
  - Expected perplexity: ~50-60
- `medium` - ~117M parameters (d_model=768, num_heads=12, num_layers=12, d_ff=3072, max_seq_len=1024)
  - Option C: GPT-2 Small equivalent
  - Expected perplexity: ~35-40

**Default:** None (uses manual configuration)

**Example:**
```bash
./run/run_train_lm.sh --preset small
```

#### --dataset DATASET
**Objective:** Select which text corpus to train on.

**Available Values:**
- `wikitext` - WikiText-2 dataset (~2M tokens)
  - Smaller dataset, good for testing and quick experiments
  - Trains faster but lower final performance
- `wikitext-103` - WikiText-103 dataset (~100M tokens)
  - Larger dataset for serious training
  - Better performance but longer training time

**Default:** `wikitext-103`

**Example:**
```bash
./run/run_train_lm.sh --dataset wikitext
```

#### --epochs N
**Objective:** Control how many complete passes through the dataset to perform.

**Available Values:** Any positive integer (typically 5-50)
- Lower values (5-10): Quick training, may underfit
- Medium values (15-25): Standard training
- Higher values (30-50): Extended training, risk of overfitting

**Default:** `20`

**Example:**
```bash
./run/run_train_lm.sh --epochs 30
```

#### --batch_size N
**Objective:** Set the number of sequences processed together in one gradient update.

**Available Values:** Power of 2 recommended (8, 16, 32, 64, 128)
- Smaller batches (8-16): Less memory, noisier gradients
- Medium batches (32): Good balance (default)
- Larger batches (64-128): More memory, stabler gradients

**Default:** `32`

**Memory Note:** Reduce if encountering OOM (Out Of Memory) errors.

**Example:**
```bash
./run/run_train_lm.sh --batch_size 64
```

#### --lr RATE
**Objective:** Set the learning rate controlling optimization step size.

**Available Values:** Scientific notation (e.g., 3e-4, 6e-4, 1e-3)
- Smaller values (1e-5 to 1e-4): Safer but slower training
- Medium values (3e-4 to 6e-4): Standard range
- Larger values (1e-3 and above): Faster but potentially unstable

**Default:** `3e-4`

**Example:**
```bash
./run/run_train_lm.sh --lr 6e-4
```

#### --warmup STEPS
**Objective:** Configure learning rate warmup to stabilize early training.

**Available Values:** 0 (disabled) or positive integer
- `0`: No warmup (not recommended for large models)
- `1000-2000`: Standard warmup for small models
- `2000-4000`: Recommended for medium/large models
- `4000+`: Extended warmup for very large models

**Default:** `0` (in shell script, but train_lm.py defaults to 1000)

**How it works:** LR linearly increases from 0 to peak over warmup steps, then cosine decays.

**Example:**
```bash
./run/run_train_lm.sh --preset small --warmup 4000 --lr 6e-4
```

#### --snapshot N
**Objective:** Control how frequently to save intermediate checkpoints during training.

**Available Values:**
- `0`: Disabled (only save per-epoch snapshots)
- Positive integer: Save every N training steps

**Default:** `10000`

**Use Cases:**
- Long epochs: Use smaller values (1000-5000) for more frequent checkpoints
- Short epochs: Use 0 or large values (10000+)

**Example:**
```bash
./run/run_train_lm.sh --snapshot 5000
```

#### --monitor_interval N
**Objective:** Set frequency for gradient and weight monitoring checks.

**Available Values:** Positive integer (steps)
- Smaller values (100-1000): More frequent monitoring, more overhead
- Larger values (5000-10000): Less frequent, lower overhead

**Default:** `10000`

**Example:**
```bash
./run/run_train_lm.sh --monitor_interval 5000
```

#### --sanity_check_interval N
**Objective:** Configure validation checks for NaN/Inf in weights and gradients.

**Available Values:**
- `0`: Disabled
- Positive integer: Check every N steps

**Default:** `10000`

**Example:**
```bash
./run/run_train_lm.sh --sanity_check_interval 1000
```

#### --help, -h
**Objective:** Display usage information.

**Example:**
```bash
./run/run_train_lm.sh --help
```

## Model Presets

### Preset Comparison Table

| Preset   | Parameters | d_model | Heads | Layers | d_ff | Context | Dataset      | Expected PPL |
|----------|-----------|---------|-------|--------|------|---------|--------------|--------------|
| `tiny`   | ~16M      | 256     | 4     | 4      | 1024 | 256     | WikiText-2   | ~80-100      |
| `small`  | ~45-50M   | 512     | 8     | 6      | 2048 | 512     | WikiText-103 | ~50-60       |
| `medium` | ~117M     | 768     | 12    | 12     | 3072 | 1024    | WikiText-103 | ~35-40       |

### Preset Usage Examples

```bash
# Tiny model for quick testing
./run/run_train_lm.sh --preset tiny --dataset wikitext --epochs 10

# Small model for standard training (Option A)
./run/run_train_lm.sh --preset small --warmup 4000 --lr 6e-4

# Medium model (GPT-2 Small equivalent)
./run/run_train_lm.sh --preset medium --warmup 4000 --batch_size 16
```

## Advanced Usage

### Direct Python Script Usage

For full control over all training parameters, call the Python script directly:

```bash
# Set PYTHONPATH
export PYTHONPATH="/home/masa/repository/python-programs/transformer/src"

# Run training with custom configuration
python src/training/train_lm.py \
    --dataset wikitext-103 \
    --model_preset small \
    --epochs 30 \
    --batch_size 32 \
    --lr 6e-4 \
    --warmup_steps 4000 \
    --gradient_clip 1.0 \
    --weight_decay 0.1 \
    --snapshot_interval 5000 \
    --gradient_monitor \
    --gradient_monitor_interval 1000 \
    --early_stopping \
    --early_stop_patience 5 \
    --weight_monitor \
    --weight_monitor_interval 100
```

### Additional Python Script Options

The Python script provides many additional options not exposed in the shell wrapper:

#### Data Options
- `--tokenizer {gpt2,gpt4,gpt4o}` - Tokenizer selection (default: gpt2)
- `--max_samples N` - Limit training samples for quick testing

#### Advanced Training Options
- `--weight_decay RATE` - L2 regularization (default: 0.1)
- `--min_lr_ratio RATIO` - Min LR as fraction of peak (default: 0.1)
- `--gradient_clip MAX_NORM` - Gradient clipping threshold (default: 1.0)
- `--resume` - Resume from latest checkpoint
- `--checkpoint_file PATH` - Resume from specific checkpoint
- `--yes, -y` - Auto-confirm checkpoint loading

#### Snapshot Management
- `--keep_last_n_snapshots N` - Number of snapshots to retain (default: 3)
- `--delete_snapshots_after_training` - Clean up snapshots when done (default: enabled)
- `--no_delete_snapshots_after_training` - Keep all snapshots

#### Monitoring Options
- `--gradient_monitor` - Enable gradient flow monitoring
- `--gradient_monitor_interval N` - Monitor every N steps
- `--weight_monitor` - Enable weight update monitoring (default: enabled)
- `--weight_monitor_interval N` - Monitor every N steps (default: 100)
- `--weight_monitor_sample_size SIZE` - Sample size for monitoring (default: 1024)

#### Early Stopping Options
- `--early_stopping` - Enable early stopping
- `--early_stop_patience N` - Epochs without improvement before stopping (default: 5)
- `--early_stop_min_delta DELTA` - Minimum improvement to count (default: 0.001)
- `--no_early_stop_restore_best` - Don't restore best weights
- `--early_stop_overfit_patience N` - Stop if val-train gap grows for N epochs
- `--early_stop_overfit_min_delta DELTA` - Min gap increase to count (default: 0.01)

#### Model Architecture Options
- `--d_model DIM` - Model dimension (default: 256)
- `--num_heads N` - Number of attention heads (default: 4)
- `--num_layers N` - Number of decoder layers (default: 4)
- `--d_ff DIM` - Feed-forward dimension (default: 512)
- `--max_seq_len LEN` - Maximum sequence length (default: 256)
- `--dropout RATE` - Dropout rate (default: 0.1)

#### Information Options
- `--info` - Show comprehensive documentation
- `--explain` - Show explanation with examples

### Resume Training Examples

```bash
# Resume from latest checkpoint (interactive prompt)
./run/run_train_lm.sh --preset small

# Then when prompted, the script finds the latest checkpoint
# Or run directly with Python:
python src/training/train_lm.py --resume

# Auto-confirm checkpoint loading (non-interactive)
python src/training/train_lm.py --resume --yes

# Resume from specific checkpoint file
python src/training/train_lm.py --checkpoint_file result/lm_wikitext-103/snapshots/snapshot_epoch_10.pt
```

## Monitoring Training

### View Training Progress

The shell script runs training in the background and creates log files.

```bash
# Follow the log in real-time
tail -f result/lm_wikitext-103/logs/train_TIMESTAMP.log

# Example with actual timestamp
tail -f result/lm_wikitext-103/logs/train_2026FEB15_215234_GMT+00.log
```

### Check Process Status

```bash
# After starting training, the script prints the process ID
# Check if training is still running
ps -p PID

# Example
ps -p 12345
```

### Stop Training

```bash
# Kill the training process
kill PID

# Example
kill 12345

# Force kill if necessary
kill -9 12345
```

### TensorBoard Monitoring

```bash
# Start TensorBoard to visualize training metrics
tensorboard --logdir result/lm_wikitext-103/tensorboard

# Then open browser to http://localhost:6006
```

### Output Locations

Training outputs are organized as follows:

```
result/lm_{dataset}/
├── logs/
│   └── train_TIMESTAMP.log     # Training logs
├── snapshots/
│   ├── snapshot_epoch_*.pt     # Intermediate checkpoints
│   └── run_config.json         # Configuration for resume
├── models/
│   └── model_TIMESTAMP.pt      # Final trained model
└── tensorboard/                # TensorBoard event files
```

## Example Training Sessions

### Session 1: Quick Test
```bash
# Fast test run with tiny model on small dataset
./run/run_train_lm.sh \
    --preset tiny \
    --dataset wikitext \
    --epochs 5 \
    --batch_size 64
```

### Session 2: Standard Training (Option A)
```bash
# Balanced model for WikiText-103
./run/run_train_lm.sh \
    --preset small \
    --dataset wikitext-103 \
    --epochs 30 \
    --warmup 4000 \
    --lr 6e-4 \
    --snapshot 5000
```

### Session 3: Large Model with Full Monitoring
```bash
# GPT-2 Small equivalent with comprehensive monitoring
python src/training/train_lm.py \
    --model_preset medium \
    --dataset wikitext-103 \
    --epochs 40 \
    --batch_size 16 \
    --lr 3e-4 \
    --warmup_steps 4000 \
    --gradient_clip 1.0 \
    --snapshot_interval 2000 \
    --gradient_monitor \
    --gradient_monitor_interval 500 \
    --early_stopping \
    --early_stop_patience 7 \
    --early_stop_overfit_patience 3 \
    --weight_monitor \
    --weight_monitor_interval 100 \
    --keep_last_n_snapshots 5 \
    --yes
```

### Session 4: Resume Interrupted Training
```bash
# Resume from where it left off
python src/training/train_lm.py \
    --resume \
    --yes
```

## Best Practices

1. **Start Small**: Begin with `--preset tiny` on `wikitext` to verify everything works
2. **Use Warmup**: Always use `--warmup` for medium/large models (2000-4000 steps)
3. **Monitor Early**: Check logs frequently in the first few hundred steps
4. **Save Snapshots**: Use `--snapshot` for long-running training (every 2000-5000 steps)
5. **Enable Monitoring**: Use `--gradient_monitor` and `--weight_monitor` to catch training issues
6. **Background Execution**: The shell script uses nohup for uninterrupted training
7. **Resume Support**: Training can be resumed if interrupted using `--resume`

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `--batch_size` (try 16, then 8)
- Reduce `--max_seq_len` if using direct Python script
- Use smaller model preset

### Training Instability (Loss → NaN)
- Reduce `--lr` (try 1e-4 or 3e-5)
- Increase `--warmup` steps
- Check `--gradient_clip` is enabled (default: 1.0)

### Slow Training
- Increase `--batch_size` if memory allows
- Verify GPU is being used (check logs for "Device: cuda")
- Reduce monitoring intervals

### Training Not Improving
- Increase `--epochs`
- Adjust `--lr` (try both higher and lower)
- Try different model preset
- Increase `--warmup_steps`

## Additional Resources

- Architecture details: See source files in `src/model/`
- Trainer implementation: `src/training/trainer.py`
- Data loading: `src/training/loader.py`
- Callbacks: `src/training/trainer_*.py`
