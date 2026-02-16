# EN-ES Translation Training Guide

## Quick Start

### 1. Quick Test (Recommended First)
Test the setup with 100 samples and 1 epoch (~30 seconds):
```bash
./run_train_translation.sh --quick
```

### 2. Full Training
Train on complete OPUS Books dataset (20 epochs, ~several hours):
```bash
# Recommended: Run in tmux to prevent SSH disconnection
tmux new -s translation
./run_train_translation.sh
# Detach with: Ctrl+B, then D
# Re-attach with: tmux attach -t translation
```

### 3. Resume Training
Resume from the latest checkpoint:
```bash
./run_train_translation.sh --resume
```

---

## Usage Examples

### Basic Usage
```bash
# Default: EN→ES, 20 epochs, batch_size=32
./run_train_translation.sh

# With custom parameters
./run_train_translation.sh --epochs 10 --batch_size 16 --lr 5e-4

# Different language pair (French to English)
./run_train_translation.sh --source_language fr --target_language en

# Limit samples for testing
./run_train_translation.sh --max_samples 1000 --epochs 2
```

### Common Scenarios

**Testing/Debugging:**
```bash
./run_train_translation.sh --quick
```

**Prevent OOM (Out of Memory):**
```bash
./run_train_translation.sh --batch_size 16 --max_seq_len 64
```

**Longer sequences:**
```bash
./run_train_translation.sh --max_seq_len 256
```

**Different tokenizers:**
```bash
# GPT-4 tokenizer (larger vocab: 100K tokens)
./run_train_translation.sh --source_tokenizer gpt4 --target_tokenizer gpt4

# GPT-4o tokenizer (largest vocab: 200K tokens)
./run_train_translation.sh --source_tokenizer gpt4o --target_tokenizer gpt4o
```

---

## Command Line Options

### Data Options
| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | opus_books | Dataset to use (opus_books) |
| `--source_language` | en | Source language code (en, fr, de, es, etc.) |
| `--target_language` | es | Target language code (en, fr, de, es, etc.) |
| `--source_tokenizer` | gpt2 | Source tokenizer (gpt2, gpt4, gpt4o) |
| `--target_tokenizer` | gpt2 | Target tokenizer (gpt2, gpt4, gpt4o) |
| `--max_samples` | None | Limit training samples (for testing) |

### Training Options
| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 32 | Batch size (reduce if OOM) |
| `--lr` | 3e-4 | Learning rate |
| `--weight_decay` | 0.1 | L2 regularization |
| `--gradient_clip` | 1.0 | Gradient clipping max norm |
| `--max_seq_len` | 128 | Max sequence length |
| `--resume` | False | Resume from checkpoint |
| `--snapshot_interval` | 5000 | Save snapshot every N steps |

### Model Architecture Options
| Option | Default | Description |
|--------|---------|-------------|
| `--d_model` | 256 | Model embedding dimension |
| `--encoder_num_layers` | 4 | Number of encoder layers |
| `--decoder_num_layers` | 4 | Number of decoder layers |
| `--encoder_num_heads` | 4 | Encoder attention heads |
| `--decoder_num_heads` | 4 | Decoder attention heads |
| `--encoder_d_ff` | 512 | Encoder feed-forward dimension |
| `--decoder_d_ff` | 512 | Decoder feed-forward dimension |
| `--dropout` | 0.1 | Dropout rate |

---

## Output Structure

Training results are saved to:
```
result/translation_opus_books_en_es/
├── models/                          # Final trained models
│   └── model_YYYYMMDD_HHMMSS.pt    # PyTorch model checkpoint
├── snapshots/                       # Training checkpoints
│   ├── run_config.json              # Configuration used
│   └── snapshot_epoch_XXXX_*.pt     # Periodic snapshots
└── logs/                            # Training logs
    └── train_YYYYMMDD_HHMMSS.log   # Training progress log
```

---

## Model Architecture

The script trains an **encoder-decoder Transformer** (Vaswani et al., 2017):

```
Source Text (EN) → Encoder → Memory
                               ↓
Target Text (ES) → Decoder → Cross-Attention → Output (ES translation)
```

**Key Features:**
- ✅ Encoder self-attention with source padding mask
- ✅ Decoder causal self-attention with target padding mask
- ✅ Cross-attention between decoder and encoder
- ✅ Teacher forcing during training
- ✅ Separate tokenizers for source and target
- ✅ Positional encoding
- ✅ Layer normalization and dropout

---

## Performance Tips

### Memory Optimization
If you encounter Out of Memory (OOM) errors:

1. **Reduce batch size:**
   ```bash
   ./run_train_translation.sh --batch_size 16
   ```

2. **Reduce sequence length:**
   ```bash
   ./run_train_translation.sh --max_seq_len 64
   ```

3. **Smaller model:**
   ```bash
   ./run_train_translation.sh --d_model 128 --encoder_num_layers 2 --decoder_num_layers 2
   ```

### Training Speed
For faster training:

1. **Increase batch size** (if memory allows):
   ```bash
   ./run_train_translation.sh --batch_size 64
   ```

2. **Use GPU** (automatic if available)

3. **Reduce validation frequency** (not currently exposed in script)

### Prevent Training Interruption

**Always use tmux or screen:**
```bash
# Start tmux session
tmux new -s translation

# Run training
./run_train_translation.sh

# Detach: Ctrl+B, then D
# Re-attach: tmux attach -t translation
# List sessions: tmux ls
```

**Enable swap space** (prevents OOM kills):
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Monitoring Training

### View Training Progress
```bash
# Follow the log file
tail -f result/translation_opus_books_en_es/logs/train_*.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check system resources
htop
```

### Training Metrics
The script logs:
- Loss per step
- Train loss per epoch
- Validation loss per epoch
- EMA (Exponential Moving Average) loss

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Make sure you run the script from the transformer root directory:
```bash
cd /home/masa/repository/python-programs/transformer
./run_train_translation.sh
```

### Issue: Training stops unexpectedly
**Solutions:**
1. Check if running in tmux/screen
2. Enable swap space
3. Reduce batch size or sequence length
4. Check system logs: `dmesg -T | grep -i oom`

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size:
```bash
./run_train_translation.sh --batch_size 8
```

### Issue: Slow training
**Solutions:**
1. Check GPU is being used: `nvidia-smi`
2. Increase batch size if memory available
3. Check if other processes are using GPU

---

## Comparison with LM Training

| Feature | Translation Training | LM Training |
|---------|---------------------|-------------|
| Architecture | Encoder-decoder | Decoder-only |
| Task | EN→ES translation | Next token prediction |
| Dataset | OPUS Books (parallel) | WikiText-103 |
| Script | `run_train_translation.sh` | (command line) |
| Input | Source + target | Text only |
| Tokenizers | Dual (src + tgt) | Single |
| Attention | Encoder + decoder + cross | Decoder causal only |

---

## Next Steps

After training completes:

1. **Evaluate the model** (implement evaluation script)
2. **Test translations** (implement inference script)
3. **Fine-tune** on domain-specific data
4. **Export for deployment**

---

## Support

For issues or questions:
- Check the training log in `result/translation_opus_books_en_es/logs/`
- Review the configuration in `result/translation_opus_books_en_es/snapshots/run_config.json`
- Check GPU status with `nvidia-smi`
- Monitor resources with `htop` or `free -h`

---

**Last Updated:** 2026-02-16
