"""Module to implement Transformer

For a ChatGPT-like experience with automatic tokenization, see TransformerAPI in app.py.
"""
import logging
from typing import Optional

import torch
from torch import (
    Tensor,
    nn
)

from .constant import (
    TYPE_FLOAT,
    NUM_ENCODER_TOKENS,
    NUM_DECODER_TOKENS,
    NUM_CLASSES,
    MAX_TIME_STEPS,
    DIM_MODEL,
    DIM_PWFF_HIDDEN,
    NUM_LAYERS,
    NUM_HEADS,
    DROPOUT_RATIO,
)
from .common import (
    Projection,
)
from .encoder import (
    Encoder
)
from .decoder import (
    Decoder
)


logger: logging.Logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    """Class to implement the Google Transformer Model"""
    def __init__(self):
        super().__init__()

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder: nn.Module = Encoder(
            vocabulary_size=NUM_ENCODER_TOKENS,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            d_model=DIM_MODEL,
            dtype=TYPE_FLOAT,
            d_ff=DIM_PWFF_HIDDEN,
            do_mask=False,
            max_time_steps=MAX_TIME_STEPS,
            bias=True,
            p_drop=DROPOUT_RATIO
        )

        self.decoder: nn.Module = Decoder(
            vocabulary_size=NUM_DECODER_TOKENS,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            d_model=DIM_MODEL,
            dtype=TYPE_FLOAT,
            d_ff=DIM_PWFF_HIDDEN,
            max_time_steps=MAX_TIME_STEPS,
            bias=True,
            p_drop=DROPOUT_RATIO
        )

        self.projection: nn.Module = Projection(
            d_model=DIM_MODEL,
            num_classes=NUM_CLASSES,
            dtype=TYPE_FLOAT,
            bias=True
        )

        # --------------------------------------------------------------------------------
        # Default token IDs for autoregressive generation. They depend on your tokenizer/vocabulary.
        #
        # Example with HuggingFace GPT-2 (BPE tokenizer):
        #   from transformers import GPT2Tokenizer
        #   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #   model.start_token = tokenizer.bos_token_id  # 50256 ('<|endoftext|>')
        #   model.end_token = tokenizer.eos_token_id    # 50256 (same as bos in GPT-2)
        #
        # Example with HuggingFace BERT (WordPiece tokenizer):
        #   from transformers import BertTokenizer
        #   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #   model.start_token = tokenizer.cls_token_id  # 101 ('[CLS]')
        #   model.end_token = tokenizer.sep_token_id    # 102 ('[SEP]')
        #
        # Example with HuggingFace T5 (SentencePiece tokenizer):
        #   from transformers import T5Tokenizer
        #   tokenizer = T5Tokenizer.from_pretrained('t5-small')
        #   model.start_token = tokenizer.pad_token_id  # 0 (T5 uses pad as decoder start)
        #   model.end_token = tokenizer.eos_token_id    # 1 ('</s>')
        #
        # Usage:
        #   model.start_token = tokenizer.bos_token_id
        #   model.end_token = tokenizer.eos_token_id
        #   output = model(x)  # uses default tokens
        #
        # Or pass explicitly per call:
        #   output = model(x, start_token=101, end_token=102)
        #
        # Why do we need start_token and end_token?
        # When you use ChatGPT, you just type text and get a response - you don't see special tokens.
        # That's because the API handles them internally:
        #
        #     You type:     "Hello, how are you?"
        #
        #     API internally:
        #     1. Tokenizer adds:  [<START>] + "Hello, how are you?" + [<END>]
        #     2. Model generates: [<START>] "I'm doing well..." [<END>]
        #     3. API removes:     special tokens
        #
        #     You see:      "I'm doing well..."
        #
        # When building a Transformer from scratch, there's no wrapper - you run the raw model.
        # The decoder needs to know:
        #   - start_token: Where to START generating (signals beginning of sequence)
        #   - end_token: When to STOP generating (signals end of sequence)
        #
        # For a ChatGPT-like experience, use the TransformerAPI wrapper class.
        # --------------------------------------------------------------------------------
        self.start_token: Optional[int] = None
        self.end_token: Optional[int] = None

    def __enter__(self) -> 'Transformer':
        """Enter context manager: set model to eval mode for inference."""
        self.eval()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager: set model back to train mode."""
        self.train()
        return False

    # DO NOT override nn.Module.__call__ as it breaks expected PyTorch behavior
    # (hooks, device handling, and normal forward invocation).
    #def __call__(
    #        self,
    #        x: Tensor,
    #        start_token: Optional[int] = None,
    #        end_token: Optional[int] = None,
    #        max_length: int = MAX_TIME_STEPS
    #) -> Tensor:
    #    """Generate output sequence from source x (inference mode).
    #    For training, use forward() directly.
    #
    #    Args:
    #        x: Source sequence of shape (B, Te)
    #        start_token: Token index to start generation (uses self.start_token if None)
    #        end_token: Token index to stop generation (uses self.end_token if None)
    #        max_length: Maximum number of tokens to generate
    #
    #    Returns: Generated token indices of shape (B, Td)
    #    """
    #    start = start_token if start_token is not None else self.start_token
    #    end = end_token if end_token is not None else self.end_token
    #
    #    if start is None or end is None:
    #        raise ValueError(
    #            "start_token and end_token must be provided either as arguments "
    #            "or set via model.start_token and model.end_token"
    #        )
    #
    #    return self.generate(x=x, start_token=start, end_token=end, max_length=max_length)

    def forward(
            self,
            x: Tensor,
            y: Tensor
    ) -> Tensor:
        """Run Transformer encoder/decoder
        DO NOT place non-differentiable functions e.g, argmax in forward.
        This returns log-probabilities. Do NOT pass the Projection output to CrossEntropyLoss
        as it expects logits and internally applies log-softmax + NLLLoss.

        Args:
            x: Encoder source sequences as token indices of shape (B, Te).
            y: Decoder target sequences as token indices of shape (B, Td).

        Returns: Log probabilities of shape (B, T, V) for training with NLLLoss.
                 DO NOT pass to CrossEntropyLoss as it expects logits.
        """
        # assert x.ndim == 2, f"expected x.shape (B, T), got {x.shape}."
        # assert y.ndim == 2, f"expected y.shape (B, T), got {y.shape}."
        if x.ndim != 2:
            raise ValueError(f"expected y.shape (B, T), got {y.shape}.")
        if y.ndim != 2:
            raise ValueError(f"expected x.shape (B, T), got {x.shape}.")
        if x.device != y.device:
            raise RuntimeError(f"Source (x) is on {x.device} but Target (y) is on {y.device}")

        # Projection returns log-probabilities. Do NOT pass the Projection output to
        # CrossEntropyLoss as it expects logits and internally applies log-softmax + NLLLoss.
        log_probs: Tensor = self.projection(y=self.decoder(y=y, memory=self.encoder(x=x)))
        return log_probs

    @torch.no_grad()
    def evaluate(
            self,
            x: Tensor,
            y: Tensor
    ) -> Tensor:
        """Get predicted tokens given source x and target context y (teacher forcing).
        Used for evaluation metrics like accuracy by comparing output against ground truth.

        Args:
            x: Encoder source sequences as token indices of shape (B, Te).
            y: Decoder target sequences as token indices of shape (B, Td).
               Typically shifted right: [<START>, tok1, tok2, ...]

        Returns: Predicted token indices of shape (B, Td).
                 Typically, shifted left: [tok1, tok2, ..., <END>]
        """
        if x.device != y.device:
            raise RuntimeError(f"Source (x) is on {x.device} but Target (y) is on {y.device}")

        log_probs = self.forward(x=x, y=y)
        predictions = torch.argmax(log_probs, dim=-1)
        return predictions

    @torch.no_grad()
    def generate(
            self,
            x: Tensor,
            start_token: int,
            end_token: int,
            max_length: int = MAX_TIME_STEPS
    ) -> Tensor:
        """Autoregressively generate output sequence from source x.
        Args:
            x: Source sequence of shape (B, Te)
            start_token: Token index to start generation (e.g., <START>)
            end_token: Token index to stop generation (e.g., <END>)
            max_length: Maximum number of tokens to generate

        Returns: Generated token indices of shape (B, Td) where Td <= max_length
        """
        if x.ndim != 2:
            msg: str = f"expected x.shape (B, T), got {x.shape}."
            logger.error(msg)
            raise ValueError(msg )

        # Warn the user about device/dtype mismatches instead of moving/casting.
        if x.device != self._device():
            logger.warning(
                "Input tensor x device %s does not match model device %s. "
                "Move your inputs explicitly to the desired device.",
                x.device, self._device()
            )

        _B = x.shape[0]

        # Encode source sequence once
        memory = self.encoder(x=x)

        # Start with <START> token for each sequence in batch
        y = torch.full((_B, 1), start_token, dtype=torch.long, device=x.device)

        for _ in range(max_length - 1):
            # Get prediction for the next token
            log_probs = self.projection(y=self.decoder(y=y, memory=memory))
            next_token = torch.argmax(log_probs[:, -1, :], dim=-1, keepdim=True)

            # Append predicted token to sequence
            y = torch.cat([y, next_token], dim=1)

            # Stop if all sequences have produced end_token
            if (next_token == end_token).all():
                break

        return y
