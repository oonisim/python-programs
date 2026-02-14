"""Simple example of loading a trained model and generating text.

This is a minimal example showing the core steps to use a trained language model.
"""
import torch
from transformers import GPT2Tokenizer
from model.lm import LanguageModel


def main():
    # ============================================================================
    # STEP 1: Load the tokenizer
    # ============================================================================
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============================================================================
    # STEP 2: Load the trained model
    # ============================================================================
    checkpoint_path = "lm_wikitext/snapshots/snapshot_epoch_0000_step_070999_20260212_095134.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model configuration
    model_config = checkpoint.get('model_config', {})

    # Create model with the same architecture
    model = LanguageModel(
        vocab_size=model_config.get('vocab_size', 50257),
        d_model=model_config.get('d_model', 256),
        num_heads=model_config.get('num_heads', 4),
        num_layers=model_config.get('num_layers', 4),
        d_ff=model_config.get('d_ff', 512),
        max_seq_len=model_config.get('max_seq_len', 256),
        dropout=model_config.get('dropout', 0.1)
    )

    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Set the end token (for stopping generation)
    model.end_token = tokenizer.eos_token_id

    print(f"Model loaded! Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ============================================================================
    # STEP 3: Generate text from a prompt
    # ============================================================================
    prompt = "The history of artificial intelligence"
    print(f"\nPrompt: {prompt}")
    print("\nGenerating...\n")

    # Tokenize the prompt
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    # Generate with the model
    with torch.no_grad():
        with model:  # This sets the model to eval mode
            generated_ids = model.generate(
                prompt=prompt_tensor,
                max_length=100,      # Total length including prompt
                temperature=0.8,     # Lower = more deterministic, higher = more random
                top_k=50,           # Keep only top 50 tokens
                top_p=0.9           # Nucleus sampling
            )

    # Decode the generated token IDs back to text
    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)

    # ============================================================================
    # STEP 4: Display the result
    # ============================================================================
    print("="*60)
    print("Generated Text:")
    print("="*60)
    print(generated_text)
    print("="*60)


if __name__ == "__main__":
    main()
