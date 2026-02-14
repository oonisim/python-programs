# Language Model Generation Guide

This guide explains how to use your trained language model (from `train_lm.py`) for text generation.

## Table of Contents
- [Quick Start](#quick-start)
- [Using the Generation Script](#using-the-generation-script)
- [Programmatic Usage](#programmatic-usage)
- [Generation Parameters](#generation-parameters)
- [Tips and Best Practices](#tips-and-best-practices)

---

## Quick Start

### 1. Find Your Trained Model

After training with `train_lm.py`, your model checkpoints are saved in:
```
lm_<dataset>/snapshots/snapshot_*.pt
```

For example:
```bash
lm_wikitext/snapshots/snapshot_epoch_0000_step_070999_20260212_095134.pt
```

### 2. Generate Text (Command Line)

**Single prompt generation:**
```bash
python generate_lm.py \
    --checkpoint lm_wikitext/snapshots/snapshot_epoch_0000_step_070999_20260212_095134.pt \
    --prompt "The history of artificial intelligence" \
    --max_length 100 \
    --temperature 0.8 \
    --top_p 0.9
```

**Interactive mode:**
```bash
python generate_lm.py \
    --checkpoint lm_wikitext/snapshots/snapshot_epoch_0000_step_070999_20260212_095134.pt \
    --interactive
```

### 3. Generate Text (Python Code)

See `example_generate.py` for a complete working example, or use this minimal code:

```python
import torch
from transformers import GPT2Tokenizer
from model.lm import LanguageModel

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load model
checkpoint = torch.load("path/to/checkpoint.pt")
model_config = checkpoint['model_config']

model = LanguageModel(
    vocab_size=model_config['vocab_size'],
    d_model=model_config['d_model'],
    num_heads=model_config['num_heads'],
    num_layers=model_config['num_layers'],
    d_ff=model_config['d_ff'],
    max_seq_len=model_config['max_seq_len'],
    dropout=model_config['dropout']
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.end_token = tokenizer.eos_token_id

# Generate
prompt_ids = torch.tensor(tokenizer.encode("Once upon a time"))
with model:
    output_ids = model.generate(prompt_ids, max_length=100, temperature=0.8)

# Decode
text = tokenizer.decode(output_ids[0].tolist())
print(text)
```

---

## Using the Generation Script

### Command-Line Arguments

**Required:**
- `--checkpoint PATH`: Path to the trained model checkpoint (.pt file)

**Generation Mode (choose one):**
- `--prompt TEXT`: Generate from a single prompt
- `--interactive`: Interactive mode (enter multiple prompts)

**Generation Parameters (optional):**
- `--max_length N`: Maximum sequence length (default: 100)
- `--temperature FLOAT`: Sampling temperature (default: 1.0)
- `--top_k N`: Top-k filtering (default: None)
- `--top_p FLOAT`: Nucleus sampling threshold (default: None)
- `--device STR`: Device to use, 'cuda' or 'cpu' (default: auto)
- `--tokenizer STR`: Tokenizer name (default: gpt2)

### Examples

**Basic generation:**
```bash
python generate_lm.py \
    --checkpoint lm_wikitext/snapshots/latest.pt \
    --prompt "In the year 2050,"
```

**More creative (higher temperature):**
```bash
python generate_lm.py \
    --checkpoint lm_wikitext/snapshots/latest.pt \
    --prompt "Once upon a time" \
    --temperature 1.5 \
    --max_length 200
```

**More deterministic (lower temperature):**
```bash
python generate_lm.py \
    --checkpoint lm_wikitext/snapshots/latest.pt \
    --prompt "The capital of France is" \
    --temperature 0.3 \
    --top_k 10
```

**Balanced generation (recommended starting point):**
```bash
python generate_lm.py \
    --checkpoint lm_wikitext/snapshots/latest.pt \
    --prompt "Artificial intelligence is" \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_length 150
```

**Interactive exploration:**
```bash
python generate_lm.py \
    --checkpoint lm_wikitext/snapshots/latest.pt \
    --interactive \
    --temperature 0.8 \
    --top_p 0.9
```

---

## Programmatic Usage

### Full Example with Error Handling

```python
import torch
from pathlib import Path
from transformers import GPT2Tokenizer
from model.lm import LanguageModel


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('model_config', {})

    # Create model
    model = LanguageModel(
        vocab_size=model_config.get('vocab_size', 50257),
        d_model=model_config.get('d_model', 256),
        num_heads=model_config.get('num_heads', 4),
        num_layers=model_config.get('num_layers', 4),
        d_ff=model_config.get('d_ff', 512),
        max_seq_len=model_config.get('max_seq_len', 256),
        dropout=model_config.get('dropout', 0.1)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def generate(model, tokenizer, prompt, max_length=100, temperature=1.0, top_p=0.9):
    """Generate text from a prompt."""
    # Tokenize
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=next(model.parameters()).device)

    # Set end token
    model.end_token = tokenizer.eos_token_id

    # Generate
    with torch.no_grad():
        with model:
            output_ids = model.generate(
                prompt=prompt_tensor,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )

    # Decode
    return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)


# Usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model, checkpoint = load_model("lm_wikitext/snapshots/latest.pt", device)

    # Generate
    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="The future of technology",
        max_length=100,
        temperature=0.8,
        top_p=0.9
    )

    print(text)
```

### Batch Generation

Generate multiple sequences in parallel:

```python
import torch
from transformers import GPT2Tokenizer
from model.lm import LanguageModel

# ... (load model as above)

# Multiple prompts
prompts = [
    "The history of computers",
    "Once upon a time",
    "In the future, we will"
]

# Tokenize all prompts
prompt_ids_list = [tokenizer.encode(p) for p in prompts]

# Pad to same length (required for batching)
max_prompt_len = max(len(ids) for ids in prompt_ids_list)
padded_prompts = []
for ids in prompt_ids_list:
    padded = ids + [tokenizer.pad_token_id] * (max_prompt_len - len(ids))
    padded_prompts.append(padded)

# Create batch tensor
prompt_batch = torch.tensor(padded_prompts, dtype=torch.long, device=device)

# Generate for all prompts at once
model.end_token = tokenizer.eos_token_id
with torch.no_grad():
    with model:
        output_batch = model.generate(
            prompt=prompt_batch,
            max_length=100,
            temperature=0.8,
            top_p=0.9
        )

# Decode each sequence
for i, output_ids in enumerate(output_batch):
    text = tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)
    print(f"\nPrompt {i+1}: {prompts[i]}")
    print(f"Generated: {text}")
```

---

## Generation Parameters

### Temperature
Controls randomness in generation.

- **Low (0.1 - 0.5)**: Deterministic, conservative
  - Use for: factual completion, code generation
  - Example: "The capital of France is" → "Paris"

- **Medium (0.6 - 1.0)**: Balanced
  - Use for: general text generation, reasonable creativity
  - Example: Good for most use cases

- **High (1.1 - 2.0)**: Creative, random
  - Use for: creative writing, exploring unusual completions
  - Example: More diverse but potentially less coherent

```python
# Very deterministic
output = model.generate(prompt, temperature=0.3)

# Balanced (default)
output = model.generate(prompt, temperature=1.0)

# Very creative
output = model.generate(prompt, temperature=1.8)
```

### Top-K Sampling
Keep only the K most likely tokens at each step.

- Prevents sampling from extremely unlikely tokens
- Typical values: 20-100
- Lower K = more focused, higher K = more diverse

```python
# Only consider top 50 tokens
output = model.generate(prompt, top_k=50)

# More conservative (top 10)
output = model.generate(prompt, top_k=10)
```

### Top-P (Nucleus) Sampling
Keep tokens until cumulative probability reaches P.

- **Adaptive**: Adjusts vocabulary size based on confidence
- Typical values: 0.9-0.95
- Lower P = more focused, higher P = more diverse

```python
# Keep tokens comprising 90% of probability mass
output = model.generate(prompt, top_p=0.9)

# More conservative
output = model.generate(prompt, top_p=0.8)

# More diverse
output = model.generate(prompt, top_p=0.95)
```

### Max Length
Maximum total tokens (prompt + generated).

- Model has a maximum context window (e.g., 256 tokens)
- If max_length exceeds context, model uses sliding window
- Longer sequences take more time to generate

```python
# Short completion
output = model.generate(prompt, max_length=50)

# Longer text
output = model.generate(prompt, max_length=200)
```

### Recommended Combinations

**Creative writing:**
```python
output = model.generate(
    prompt="Once upon a time",
    max_length=200,
    temperature=1.0,
    top_p=0.95
)
```

**Factual completion:**
```python
output = model.generate(
    prompt="The capital of France",
    max_length=50,
    temperature=0.3,
    top_k=10
)
```

**Balanced general use:**
```python
output = model.generate(
    prompt="Your prompt here",
    max_length=100,
    temperature=0.8,
    top_p=0.9
)
```

---

## Tips and Best Practices

### 1. Model Checkpoint Selection

- Use the **latest checkpoint** for best results (highest step number)
- Check validation loss in checkpoint metadata
- Lower validation loss = better model

```bash
# Find the best checkpoint
ls -lt lm_wikitext/snapshots/ | head -5
```

### 2. Prompt Engineering

**Good prompts:**
- Clear and specific: "The history of computers began in"
- Natural language: "In the year 2050, technology will"
- Match training data style: If trained on Wikipedia, use encyclopedic style

**Poor prompts:**
- Too vague: "Things"
- Unusual formatting: "###PROMPT###"
- Out-of-distribution: Emojis if model wasn't trained on them

### 3. Generation Quality

**If output is too repetitive:**
- Increase temperature (e.g., 0.8 → 1.2)
- Increase top_p (e.g., 0.9 → 0.95)
- Decrease top_k or remove it

**If output is incoherent:**
- Decrease temperature (e.g., 1.5 → 0.8)
- Decrease top_p (e.g., 0.95 → 0.85)
- Add top_k filtering (e.g., top_k=50)

**If output is boring:**
- Increase temperature
- Try different prompts
- Check if model was trained enough (perplexity should be < 30-50)

### 4. Performance Optimization

**For faster generation:**
```python
# Use CPU if GPU transfer overhead is high for short sequences
device = "cpu"

# Use smaller max_length
max_length = 50  # instead of 200

# Disable gradient computation (already done with torch.no_grad())
with torch.no_grad():
    output = model.generate(...)
```

**For batch processing:**
- Generate multiple sequences in parallel
- Pad prompts to same length
- Use larger batch sizes if memory allows

### 5. Memory Management

```python
# Clear CUDA cache if needed
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Move model to CPU when not generating
model = model.cpu()

# Load model on-demand
with torch.no_grad():
    model = model.to("cuda")
    output = model.generate(...)
    model = model.cpu()
```

### 6. Troubleshooting

**RuntimeError: Device mismatch**
```python
# Ensure prompt is on same device as model
device = next(model.parameters()).device
prompt_tensor = prompt_tensor.to(device)
```

**ValueError: Sequence too long**
```python
# Truncate prompt if too long
max_prompt_len = model.max_seq_len - 50  # Leave room for generation
prompt_ids = prompt_ids[:max_prompt_len]
```

**Poor quality output**
- Check training loss/perplexity
- Train for more epochs
- Try different generation parameters
- Use better prompts

---

## Additional Resources

### Model Architecture
- See `lm.py` for model implementation
- Decoder-only Transformer (GPT-style)
- Causal self-attention for autoregressive generation

### Training
- See `train_lm.py` for training details
- Use `--help` for all training options
- Monitor perplexity during training (lower is better)

### Testing
- See `test_lm.py` for unit tests
- Validates forward pass and generation shapes

---

## Example Use Cases

### 1. Text Completion
```python
prompt = "The three laws of robotics are"
output = model.generate(prompt, max_length=150, temperature=0.5)
```

### 2. Creative Writing
```python
prompt = "In a distant galaxy"
output = model.generate(prompt, max_length=300, temperature=1.2, top_p=0.95)
```

### 3. Question Answering (if trained on Q&A data)
```python
prompt = "Q: What is machine learning?\nA:"
output = model.generate(prompt, max_length=100, temperature=0.7)
```

### 4. Style Transfer
```python
prompt = "Rewrite in simple English: Quantum entanglement"
output = model.generate(prompt, max_length=100, temperature=0.8)
```

---

## Next Steps

1. **Experiment**: Try different prompts and parameters
2. **Evaluate**: Check if outputs match your expectations
3. **Iterate**: Retrain with more data or different settings if needed
4. **Deploy**: Integrate into your application

For more help, see:
- `python generate_lm.py --help`
- `python training/train_lm.py --explain`
- `example_generate.py` for code examples
