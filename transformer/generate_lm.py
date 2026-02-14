"""Text Generation Script for Trained Language Model.

This script loads a trained language model checkpoint and generates text
from user-provided prompts.

Usage Examples:
    # Generate from a prompt
    python generate_lm.py --checkpoint lm_wikitext/snapshots/snapshot_epoch_0000_step_070999_20260212_095134.pt --prompt "The capital of France is"

    # Interactive mode
    python generate_lm.py --checkpoint lm_wikitext/snapshots/snapshot_epoch_0000_step_070999_20260212_095134.pt --interactive

    # Control generation parameters
    python generate_lm.py --checkpoint path/to/model.pt --prompt "Once upon a time" --max_length 100 --temperature 0.8 --top_p 0.9
"""
import argparse
import torch
from pathlib import Path
from transformers import GPT2Tokenizer

from lm import LanguageModel


def load_model(checkpoint_path: str, device: str = "cuda") -> tuple[LanguageModel, dict]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model configuration from checkpoint
    model_config = checkpoint.get('model_config', {})

    # Create model with same architecture as trained model
    model = LanguageModel(
        vocab_size=model_config.get('vocab_size', 50257),
        d_model=model_config.get('d_model', 256),
        num_heads=model_config.get('num_heads', 4),
        num_layers=model_config.get('num_layers', 4),
        d_ff=model_config.get('d_ff', 512),
        max_seq_len=model_config.get('max_seq_len', 256),
        dropout=model_config.get('dropout', 0.1)
    )

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, checkpoint


def generate_text(
    model: LanguageModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = "cuda"
) -> str:
    """Generate text from a prompt.

    Args:
        model: Trained LanguageModel
        tokenizer: GPT2Tokenizer
        prompt: Text prompt to start generation
        max_length: Maximum total sequence length (prompt + generated)
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top-k tokens (None = no filtering)
        top_p: Nucleus sampling threshold (None = no filtering)
        device: Device model is on

    Returns:
        Generated text string
    """
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    # Set end token for the model
    model.end_token = tokenizer.eos_token_id

    # Generate
    with torch.no_grad():
        with model:  # Sets model to eval mode
            generated_ids = model.generate(
                prompt=prompt_tensor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)

    return generated_text


def interactive_mode(model: LanguageModel, tokenizer: GPT2Tokenizer, args):
    """Interactive text generation mode."""
    print("\n" + "="*60)
    print("Interactive Generation Mode")
    print("="*60)
    print("Type your prompt and press Enter to generate.")
    print("Type 'quit' or 'exit' to stop.")
    print(f"Settings: max_length={args.max_length}, temp={args.temperature}, "
          f"top_k={args.top_k}, top_p={args.top_p}")
    print("="*60 + "\n")

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not prompt:
                continue

            print("\nGenerating...")
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device
            )

            print("\n" + "-"*60)
            print("Generated Text:")
            print("-"*60)
            print(generated_text)
            print("-"*60 + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained language model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt to generate from (required if not using --interactive)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode (prompt repeatedly)"
    )
    parser.add_argument(
        "--max_length", type=int, default=100,
        help="Maximum total sequence length (prompt + generated). Default: 100"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature. Higher = more random. Default: 1.0"
    )
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="Keep only top-k most likely tokens. Default: None (disabled)"
    )
    parser.add_argument(
        "--top_p", type=float, default=None,
        help="Nucleus sampling threshold (0-1). Default: None (disabled)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on ('cuda' or 'cpu'). Default: auto-detect"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2",
        choices=["gpt2"],
        help="Tokenizer to use. Default: gpt2"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.interactive and args.prompt is None:
        parser.error("Either --prompt or --interactive must be specified")

    # Load tokenizer
    print(f"Loading tokenizer ({args.tokenizer})...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Load model
    model, checkpoint = load_model(args.checkpoint, args.device)

    # Run generation
    if args.interactive:
        interactive_mode(model, tokenizer, args)
    else:
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...")

        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )

        print("\n" + "="*60)
        print("Generated Text:")
        print("="*60)
        print(generated_text)
        print("="*60)


if __name__ == "__main__":
    main()
