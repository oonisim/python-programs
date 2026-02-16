# Dataset Chunking for Language Model Training

## Overview

When implementing a dataset for GPT-style language model training, the way you chunk your token sequence into training samples has a massive impact on training efficiency. This document explains the correct approach (block chunking with stride=seq_len) and a common mistake (sliding window with stride=1) that can make training 512x slower.

---

## What to Consider

### Two Approaches to Sequence Generation

Given a stream of tokens, there are two main ways to create training sequences:

#### Sliding Window (stride=1)
```python
# WRONG for training - only use for evaluation!
tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]

sequence[0] = tokens[0:4]   # [0, 1, 2, 3] → target [1, 2, 3, 4]
sequence[1] = tokens[1:5]   # [1, 2, 3, 4] → target [2, 3, 4, 5]
sequence[2] = tokens[2:6]   # [2, 3, 4, 5] → target [3, 4, 5, 6]
...
```

**Properties:**
- Creates **~N sequences** (one per token position)
- High redundancy: consecutive sequences share (seq_len - 1) tokens
- **Use case:** Evaluation only (perplexity, fine-grained metrics)
- **Never for training** - massively inefficient

#### Block Chunking (stride=seq_len)
```python
# CORRECT for training - standard practice
tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]

sequence[0] = tokens[0:4]    # [0, 1, 2, 3] → target [1, 2, 3, 4]
sequence[1] = tokens[4:8]    # [4, 5, 6, 7] → target [5, 6, 7, 8]
sequence[2] = tokens[8:12]   # [8, 9, 10, 11] → target [9, 10, 11, 12]
...
```

**Properties:**
- Creates **~N/seq_len sequences** (non-overlapping blocks)
- No redundancy: each token appears exactly once per epoch
- **Use case:** Training (GPT-2, GPT-3, all modern LMs)
- **Standard practice** - efficient, optimal data usage

### Why Block Chunking?

1. **Efficiency:** Process each token exactly once per epoch
2. **Speed:** 512x fewer steps for seq_len=512
3. **Standard:** Used by all modern language models (GPT-2, GPT-3, etc.)
4. **No correlation issues:** Adjacent batches come from different text regions

---

## How to Implement

### Correct Implementation

```python
class LanguageModelDataset(Dataset):
    """Dataset with non-overlapping block chunking (stride = seq_len)."""

    def __init__(self, tokens: Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        """Calculate number of non-overlapping chunks.

        Each sample needs seq_len + 1 tokens:
          - input:  tokens[start : start + seq_len]
          - target: tokens[start + 1 : start + seq_len + 1]

        Valid start positions: start + seq_len + 1 <= len(tokens)
        Max valid start: max_start = len(tokens) - (seq_len + 1)
        Number of chunks: (max_start // seq_len) + 1
        """
        n = len(self.tokens)
        max_start = n - (self.seq_len + 1)
        if max_start < 0:
            return 0
        return (max_start // self.seq_len) + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get non-overlapping chunk at index idx.

        Args:
            idx: Chunk index (NOT token position)

        Key: multiply idx by seq_len to get starting position
        """
        start = idx * self.seq_len  # This creates stride=seq_len
        input_ids = self.tokens[start : start + self.seq_len]
        target_ids = self.tokens[start + 1 : start + self.seq_len + 1]
        return input_ids, target_ids
```

### Key Implementation Details

**1. The `__len__` formula:**
```python
max_start = len(tokens) - (seq_len + 1)
num_chunks = (max_start // seq_len) + 1
```

**Why seq_len + 1?** Because each chunk needs:
- `seq_len` tokens for input
- `seq_len` tokens for target (shifted by 1)
- Total span: `seq_len + 1` tokens

**2. The `__getitem__` indexing:**
```python
start = idx * self.seq_len  # NOT just idx!
```

This is the critical line that creates non-overlapping chunks:
- `idx=0` → `start=0` → tokens[0:seq_len]
- `idx=1` → `start=seq_len` → tokens[seq_len:2*seq_len]
- `idx=2` → `start=2*seq_len` → tokens[2*seq_len:3*seq_len]

---

## What Will Happen (Expected Behavior)

### Example: 1025 Tokens, seq_len=256

**Dataset size calculation:**
```python
max_start = 1025 - 257 = 768
num_chunks = (768 // 256) + 1 = 3 + 1 = 4
```

**Chunks generated:**
```
Chunk 0: tokens[0:256]     → targets[1:257]
Chunk 1: tokens[256:512]   → targets[257:513]
Chunk 2: tokens[512:768]   → targets[513:769]
Chunk 3: tokens[768:1024]  → targets[769:1025]
```

**Properties:**
- 4 non-overlapping sequences
- Each token appears in exactly one input sequence
- No wasted tokens (except last few if not evenly divisible)

### Training Metrics

For WikiText-103 (~118M tokens, seq_len=512, batch_size=32):

```
Dataset size: ~230,313 sequences
Steps per epoch: 230,313 / 32 ≈ 7,197 steps
Epoch time: ~1.5 hours (at 4,740 steps/hour)
```

**Sanity check formula:**
```
expected_sequences ≈ total_tokens / seq_len
expected_steps ≈ expected_sequences / batch_size
```

If your actual steps/epoch is much higher than this, you likely have a stride bug.

---

## What Occurred (The Bug)

### Wrong Implementation (Stride=1)

```python
def __len__(self) -> int:
    """WRONG: Creates sliding window with stride=1"""
    return max(0, len(self.tokens) - self.seq_len)

def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
    """WRONG: Uses idx directly as token position"""
    input_ids = self.tokens[idx : idx + self.seq_len]
    target_ids = self.tokens[idx + 1 : idx + self.seq_len + 1]
    return input_ids, target_ids
```

### What This Creates

For 1025 tokens, seq_len=256:
```
Number of sequences: 1025 - 256 = 769

Chunk 0:   tokens[0:256]     → massive overlap
Chunk 1:   tokens[1:257]     ↓
Chunk 2:   tokens[2:258]     ↓
...
Chunk 768: tokens[768:1024]

Each consecutive pair shares 255/256 tokens (99.6% overlap!)
```

### Impact

For WikiText-103:
```
Buggy dataset size: ~117,920,000 sequences (512x too large!)
Steps per epoch: ~3,685,000 steps
Epoch time: ~777 hours = 32 DAYS
```

**Warning signs:**
1. Steps per epoch ≈ total_tokens (instead of total_tokens/seq_len)
2. Training taking days/weeks instead of hours
3. Epoch progress incredibly slow

---

## How It Was Fixed

### Changes Made

**1. Fixed `__len__` calculation:**
```python
# Before (WRONG):
return max(0, len(self.tokens) - self.seq_len)

# After (CORRECT):
max_start = len(self.tokens) - (self.seq_len + 1)
if max_start < 0:
    return 0
return (max_start // self.seq_len) + 1
```

**2. Fixed `__getitem__` indexing:**
```python
# Before (WRONG):
start = idx

# After (CORRECT):
start = idx * self.seq_len
```

**3. Updated `get_stats()` consistency:**
```python
# Calculate sequence count using same formula as __len__
max_start = len(tokens) - (seq_len + 1)
num_sequences = 0 if max_start < 0 else (max_start // seq_len) + 1
```

### Verification

Added comprehensive tests to ensure:
1. Non-overlapping chunks (stride = seq_len)
2. No token duplication across sequences
3. Correct sequence count for various input sizes
4. Edge cases (minimum tokens, exact multiples, etc.)

**Test verification:**
```python
def test_language_model_dataset_stride_non_overlapping():
    tokens = torch.arange(0, 20)
    dataset = LanguageModelDataset(tokens, seq_len=4)

    # Verify 4 non-overlapping chunks
    assert len(dataset) == 4

    # Check each chunk starts at idx * seq_len
    assert dataset[0][0].tolist() == [0, 1, 2, 3]
    assert dataset[1][0].tolist() == [4, 5, 6, 7]
    assert dataset[2][0].tolist() == [8, 9, 10, 11]
    assert dataset[3][0].tolist() == [12, 13, 14, 15]
```

### Results

```
Before Fix:
  - Dataset: ~118M sequences
  - Steps/epoch: ~3.68M
  - Time/epoch: ~32 days

After Fix:
  - Dataset: ~230K sequences
  - Steps/epoch: ~7.2K
  - Time/epoch: ~1.5 hours

Speedup: 512x faster! ✓
```

---

## Implementation Checklist

When implementing a language model dataset:

### Must Do:
- [ ] Use `start = idx * seq_len` in `__getitem__` (creates stride=seq_len)
- [ ] Calculate `__len__` as `(max_start // seq_len) + 1`
- [ ] Account for `seq_len + 1` tokens needed per sample
- [ ] Test with small examples to verify non-overlapping chunks
- [ ] Verify steps/epoch ≈ total_tokens / (seq_len × batch_size)

### Must Not Do:
- [ ] ~~Use `start = idx` (creates stride=1 sliding window)~~
- [ ] ~~Calculate `__len__` as `len(tokens) - seq_len`~~
- [ ] ~~Assume "it will be fast enough" without checking~~

### Testing:
- [ ] Verify no overlap: collect all input tokens, check for duplicates
- [ ] Test edge cases: minimum tokens, exact multiples, too few tokens
- [ ] Validate target offset: targets should be inputs + 1
- [ ] Confirm dataset size matches expected: ~N/seq_len

---

## Quick Reference

### Formulas

**Number of chunks:**
```python
max_start = len(tokens) - (seq_len + 1)
num_chunks = (max_start // seq_len) + 1 if max_start >= 0 else 0
```

**Chunk starting position:**
```python
start = idx * seq_len  # NOT just idx!
```

**Expected steps per epoch:**
```python
steps_per_epoch ≈ (total_tokens / seq_len) / batch_size
```

### When to Use Each Approach

| Approach | Stride | Use Case | Dataset Size |
|----------|--------|----------|--------------|
| **Block Chunking** | seq_len | Training (GPT-2, GPT-3) | ~N/seq_len |
| **Sliding Window** | 1 | Evaluation only | ~N |

---

## Summary

**The golden rule:** For GPT-style language model training, always use block chunking with stride=seq_len. This creates non-overlapping sequences where each token is seen exactly once per epoch.

**Remember:** If your "steps per epoch" seems unreasonably high, you likely have a stride bug. The expected steps per epoch should be approximately `total_tokens / (seq_len × batch_size)`, not `total_tokens / batch_size`.

**Standard practice:** All modern language models (GPT-2, GPT-3, etc.) use block chunking for training. Sliding window (stride=1) is only used for evaluation metrics like perplexity, never for training.
