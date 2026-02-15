# Option A Training Guide (45-50M Parameters)

GPU Memory Requirements

  Without Optimization:
  - batch_size=8: ~11 GB → RTX 3060 12GB (tight fit) or RTX 4070 12GB
  - batch_size=16: ~18.6 GB → RTX 4080 16GB (tight) or A4000 16GB
  - batch_size=32: ~33.7 GB → RTX 3090 24GB, RTX 4090 24GB, or A5000 24GB

  With Optimizations (gradient checkpointing + mixed precision fp16):
  - batch_size=8: ~4.4 GB → Trainable on RTX 3060 8GB or better
  - batch_size=16: ~7.4 GB → Trainable on RTX 3060 12GB or better
  - batch_size=32: ~13.5 GB → Trainable on RTX 4070 16GB or better

  Memory Breakdown (batch_size=8)
  ┌─────────────────────────┬────────────────────────────┐
  │        Component        │           Memory           │
  ├─────────────────────────┼────────────────────────────┤
  │ Model parameters        │ 0.50 GB                    │
  ├─────────────────────────┼────────────────────────────┤
  │ Gradients               │ 0.50 GB                    │
  ├─────────────────────────┼────────────────────────────┤
  │ Optimizer (AdamW)       │ 0.99 GB                    │
  ├─────────────────────────┼────────────────────────────┤
  │ Activations (12 layers) │ 7.55 GB ← Main bottleneck! │
  ├─────────────────────────┼────────────────────────────┤
  │ Overhead                │ 1.50 GB                    │
  ├─────────────────────────┼────────────────────────────┤
  │ TOTAL                   │ ~11 GB                     │
  └─────────────────────────┴────────────────────────────┘
  Key Insight

  The attention scores consume the most memory:
  - Each layer: (batch_size × num_heads × seq_len × seq_len) = (8 × 12 × 1024 × 1024) = 0.4 GB per layer
  - Total for 12 layers: 4.8 GB just for attention scores!

  Recommendations

  1. If you have 12GB GPU (RTX 3060, RTX 4070):
  - Use gradient checkpointing + mixed precision (fp16)
  - Start with batch_size=8 or 16
  - Should work comfortably

  2. If you have 16GB GPU (RTX 4080):
  - Can train without optimization at batch_size=8
  - Or use optimizations for batch_size=24-32

  3. If you have 24GB GPU (RTX 3090, RTX 4090):
  - Can train comfortably at batch_size=32
  - Or larger with optimizations

  Option C (117M params) is FEASIBLE on your 16GB GPU! ✓

  Recommended Configuration

  - Model: 117M parameters (GPT-2 Small equivalent)
  - Batch size: 16
  - Optimization: Mixed precision (fp16)
  - Memory usage: ~9.3 GB (58% of 16GB) - Safe!
  - Expected perplexity: ~35-40 (matching GPT-2 Small)

  Trade-offs to Consider
  ┌─────────────────────┬──────────────────────┬───────────────────┐
  │       Aspect        │    Option A (45M)    │  Option C (117M)  │
  ├─────────────────────┼──────────────────────┼───────────────────┤
  │ Parameters          │ 45-50M               │ 117M              │
  ├─────────────────────┼──────────────────────┼───────────────────┤
  │ Memory usage        │ 8-9 GB (comfortable) │ 9-13 GB (tight)   │
  ├─────────────────────┼──────────────────────┼───────────────────┤
  │ Training speed      │ Faster iterations    │ Slower iterations │
  ├─────────────────────┼──────────────────────┼───────────────────┤
  │ Expected perplexity │ 50-60                │ 35-40             │
  ├─────────────────────┼──────────────────────┼───────────────────┤
  │ Training time       │ Shorter              │ Longer            │
  ├─────────────────────┼──────────────────────┼───────────────────┤
  │ Risk                │ Very safe            │ Needs fp16        │
  └─────────────────────┴──────────────────────┴───────────────────┘


## Overview

Option A provides a balanced 45-50M parameter model suitable for WikiText-103 training on a 16GB GPU.

**Expected Performance:**
- Perplexity: ~50-60 on WikiText-103
- Much better than current 16M model (perplexity ~70)
- Comfortable memory usage on 16GB GPU

## Quick Start

### Using Model Preset (Recommended)

```bash
cd /Users/onishima/Documents/home/python-programs/transformer

# Option A with preset
python src/training/train_lm.py \
    --model_preset small \
    --dataset wikitext-103 \
    --epochs 20 \
    --batch_size 32

# With custom warmup
python src/training/train_lm.py \
    --model_preset small \
    --dataset wikitext-103 \
    --epochs 20 \
    --batch_size 32 \
    --warmup_steps 2000 \
    --lr 6e-4
```

### Manual Configuration

```bash
python src/training/train_lm.py \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --d_ff 2048 \
    --max_seq_len 512 \
    --dataset wikitext-103 \
    --epochs 20 \
    --batch_size 32 \
    --warmup_steps 1000
```

## Configuration Details

### Model Architecture (Option A)
```
d_model      = 512      # 2× current (256)
num_heads    = 8        # 2× current (4)
num_layers   = 6        # 1.5× current (4)
d_ff         = 2048     # 4× d_model (fixes previous 512)
max_seq_len  = 512      # 2× current (256)
dropout      = 0.1

Parameters: ~45-50M
```

### Training Hyperparameters
```
epochs       = 20
batch_size   = 32        # Comfortable on 16GB
learning_rate = 3e-4     # Or 6e-4 with warmup
warmup_steps = 1000      # Default (use 2000-4000 for better results)
min_lr_ratio = 0.1       # Final LR = peak_lr * 0.1
weight_decay = 0.1
gradient_clip = 1.0
```

## Memory Usage

**On 16GB GPU:**
- batch_size=32: ~8 GB (50% usage) ✓
- batch_size=48: ~11 GB (69% usage) ✓
- batch_size=64: ~14 GB (88% usage) ✓

Safe and comfortable for 16GB GPU!

## Available Model Presets

### tiny (~16M params)
Original configuration, suitable for WikiText-2
```
d_model=256, num_heads=4, num_layers=4, d_ff=1024, max_seq_len=256
```

### small (~45-50M params) **← Option A**
Balanced for WikiText-103
```
d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=512
```

### medium (~117M params)
GPT-2 Small equivalent (Option C)
```
d_model=768, num_heads=12, num_layers=12, d_ff=3072, max_seq_len=1024
```

## Training Tips

1. **Start with default settings:**
   ```bash
   python src/training/train_lm.py --model_preset small --dataset wikitext-103
   ```

2. **For faster experimentation:**
   ```bash
   python src/training/train_lm.py \
       --model_preset small \
       --dataset wikitext \  # Use WikiText-2 (smaller)
       --max_samples 10000 \
       --epochs 5
   ```

3. **For best results (more warmup):**
   ```bash
   python src/training/train_lm.py \
       --model_preset small \
       --dataset wikitext-103 \
       --warmup_steps 4000 \
       --lr 6e-4 \
       --epochs 30
   ```

4. **Resume interrupted training:**
   ```bash
   python src/training/train_lm.py --model_preset small --resume
   ```

## Key Improvements from Previous Configuration

1. **Fixed d_ff ratio**: 2048 (4× d_model) instead of 512
2. **Increased context**: max_seq_len=512 instead of 256
3. **LR warmup**: 1000 steps warmup by default
4. **Better capacity**: 45-50M params instead of 16M

## Monitoring Training

Training outputs are saved to:
```
lm_wikitext-103/
├── models/          # Final model
├── snapshots/       # Checkpoints during training
└── tensorboard/     # TensorBoard logs
```

View training progress:
```bash
tensorboard --logdir lm_wikitext-103/tensorboard
```

## Expected Training Time

On 16GB GPU with batch_size=32:
- ~4-6 hours per epoch on WikiText-103
- Total: ~80-120 hours for 20 epochs

Faster with:
- Larger batch size (if you have headroom)
- WikiText-2 instead of WikiText-103 (much smaller dataset)

## Troubleshooting

**Out of Memory:**
- Reduce `--batch_size` (try 24 or 16)
- Reduce `--max_seq_len` (try 384 or 256)

**Training too slow:**
- Increase `--batch_size` if memory allows
- Use WikiText-2 for faster iterations

**Poor perplexity:**
- Increase `--warmup_steps` to 2000-4000
- Increase `--lr` to 6e-4 with warmup
- Train for more epochs (30-40)

---

# Execution Script

* Option A - Default                                                                                                                                                                                                                                                                                               ```bash
./run_train_lm.sh --preset small
                                                       
# Option A - Enhanced warmup
./run_train_lm.sh --preset small --warmup 4000 --lr 6e-4
                                                             
# Option A - Smaller dataset for testing
./run_train_lm.sh --preset small --dataset wikitext --epochs 5
                                                 
# Tiny model for WikiText-2
./run_train_lm.sh --preset tiny --dataset wikitext
                                              
# Medium (Option C) model
./run_train_lm.sh --preset medium --warmup 2000
                                                                   
# Custom configuration (no preset)
./run_train_lm.sh --dataset wikitext-103 --epochs 30 --batch_size 48
cat /tmp/test_commands.txt
```
