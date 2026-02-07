"""High-level API wrapper for Transformer inference.

Why do we need start_token and end_token?
-----------------------------------------
When you use ChatGPT, you just type text and get a response - you don't see special tokens.
That's because the API handles them internally:

    You type:     "Hello, how are you?"

    API internally:
    1. Tokenizer adds:  [<START>] + "Hello, how are you?" + [<END>]
    2. Model generates: [<START>] "I'm doing well..." [<END>]
    3. API removes:     special tokens

    You see:      "I'm doing well..."

When building a Transformer from scratch, there's no wrapper - you run the raw model.
The decoder needs to know:
  - start_token: Where to START generating (signals beginning of sequence)
  - end_token: When to STOP generating (signals end of sequence)

This module provides TransformerAPI, a wrapper that handles tokenization and special
tokens automatically, giving you a ChatGPT-like experience.

Usage
-----
Option 1: Using TransformerAPI (recommended for inference)
    ```python
    import torch
    from transformers import T5Tokenizer
    from scratch.model import Transformer
    from scratch.app import TransformerAPI

    # Setup
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = Transformer()
    model.load_state_dict(torch.load('trained_model.pt'))

    # Create API wrapper
    api = TransformerAPI(model, tokenizer)

    # ChatGPT-like experience - just text in, text out
    result = api("Hello, how are you?")
    print(result)  # "Bonjour, comment allez-vous?"
    ```

Option 2: Using Transformer directly with context manager
    ```python
    from scratch.model import Transformer

    model = Transformer()
    model.load_state_dict(torch.load('trained_model.pt'))

    # Set default tokens once
    model.start_token = tokenizer.bos_token_id
    model.end_token = tokenizer.eos_token_id

    # Use context manager for inference
    with model:
        output = model(x)  # x is token IDs tensor of shape (B, T)
    ```

Option 3: Using Transformer for training
    ```python
    from scratch.model import Transformer

    model = Transformer()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for x, y in dataloader:
        optimizer.zero_grad()
        log_probs = model.forward(x, y)  # Returns log probs, NOT argmax
        loss = criterion(log_probs.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
    ```
"""
from .constant import MAX_TIME_STEPS
from .model import Transformer


class TransformerAPI:
    """High-level API wrapper for Transformer inference.

    Provides a ChatGPT-like experience where you just pass text and get text back,
    without dealing with tokenization or special tokens directly.

    Example:
        import torch
        from transformers import T5Tokenizer

        # Setup
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = Transformer()
        model.load_state_dict(torch.load('trained_model.pt'))

        # Create API wrapper
        api = TransformerAPI(model, tokenizer)

        # Use like ChatGPT - just text in, text out
        result = api.translate("Hello, how are you?")
        print(result)  # "Bonjour, comment allez-vous?"

        # Or even simpler with __call__:
        result = api("Hello, how are you?")
    """

    def __init__(self, model: Transformer, tokenizer):
        """Initialize the API wrapper.

        Args:
            model: Trained Transformer model
            tokenizer: HuggingFace tokenizer (or any tokenizer with encode/decode methods)
                       Must have bos_token_id and eos_token_id attributes.
        """
        self.model = model
        self.tokenizer = tokenizer

        # Configure model with tokenizer's special tokens
        self.model.start_token = getattr(tokenizer, 'bos_token_id', None) or \
                                  getattr(tokenizer, 'cls_token_id', None) or \
                                  getattr(tokenizer, 'pad_token_id', 0)
        self.model.end_token = getattr(tokenizer, 'eos_token_id', None) or \
                                getattr(tokenizer, 'sep_token_id', None) or 1

    def __call__(self, text: str, max_length: int = MAX_TIME_STEPS) -> str:
        """Translate/generate from input text.

        Args:
            text: Input text (e.g., sentence to translate)
            max_length: Maximum tokens to generate

        Returns: Generated text with special tokens removed
        """
        return self.translate(text, max_length)

    def translate(self, text: str, max_length: int = MAX_TIME_STEPS) -> str:
        """Translate/generate from input text.

        Args:
            text: Input text (e.g., sentence to translate)
            max_length: Maximum tokens to generate

        Returns: Generated text with special tokens removed
        """
        # Tokenize input text to tensor
        input_ids = self.tokenizer.encode(text, return_tensors='pt')

        # Generate output using model's context manager (sets eval mode)
        with self.model:
            output_ids = self.model(input_ids, max_length=max_length)

        # Decode back to text, removing special tokens
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text


if __name__ == "__main__":
    print(__doc__)
