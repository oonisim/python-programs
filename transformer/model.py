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

from constant import (
    TYPE_FLOAT,
    ENCODER_MAX_TIME_STEPS,
    ENCODER_MAX_TOKENS,
    ENCODER_LAYERS,
    ENCODER_PWFF_DIM,
    ENCODER_DROPOUT_RATIO,
    DECODER_MAX_TIME_STEPS,
    DECODER_MAX_TOKENS,
    DECODER_LAYERS,
    DECODER_PWFF_DIM,
    DECODER_DROPOUT_RATIO,
    DIM_MODEL,
    NUM_HEADS,
    # Number of classes to predict at the decoder output.
    # This is the same as the decoder vocabulary size.
    # NUM_CLASSES,    # pylint: disable=unused-import
)
from common import (
    InputEmbedding,
    PositionalEncoding,
    Projection,
)
from encoder import (
    Encoder
)
from decoder import (
    Decoder
)


logger: logging.Logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    """Class to implement the Google Transformer Model"""
    def __init__(
            self,
            data_type = TYPE_FLOAT,
            encoder_max_time_steps: int = ENCODER_MAX_TIME_STEPS,
            encoder_vocabulary_size: int = ENCODER_MAX_TOKENS,
            encoder_model_dimension: int = DIM_MODEL,
            encoder_pwff_dimension: int = ENCODER_PWFF_DIM,
            encoder_dropout_ratio: float = ENCODER_DROPOUT_RATIO,
            encoder_layers: int = ENCODER_LAYERS,
            decoder_max_time_steps: int = DECODER_MAX_TIME_STEPS,
            decoder_vocabulary_size: int = DECODER_MAX_TOKENS,
            decoder_model_dimension: int = DIM_MODEL,
            decoder_pwff_dimension: int = DECODER_PWFF_DIM,
            decoder_dropout_ratio: float = DECODER_DROPOUT_RATIO,
            decoder_layers: int = DECODER_LAYERS,
    ):
        """
        Note that encoder_model_dimension and decoder_model_dimension should match,
        because at the cross attention in a decoder, K and V are from encoder layer
        but Q is from decoder layer. The dimension d_model needs to match among Q,K,V.
        However, make them separate for the potential future extension to use
        separate dimensions (and adjust dimension of K,V to match Q in decoder).

        Parameters that can differ between encoder and decoder layers are:
          - d_ff (FFN hidden dimension)
          - num_layers (encoder can have 6 layers, decoder can have 12)
          - num_heads (as long as both divide d_model)
          - max_time_steps (sequence lengths are independent)
          - vocabulary_size (encoder/decoder can have different tokenizers)

        The FFN is structured as:
        x: (B, T, d_model) → W1: (d_model, d_ff) → ReLU → W2: (d_ff, d_model) → out: (B, T, d_model)
        d_ff is purely internal to each FFN block.

        Args:
            data_type: embedding vector (signal) data type
            encoder_vocabulary_size: max number of tokens in the encoder tokenizer
            encoder_max_time_steps: max sequence length T of encoder layers
            encoder_model_dimension: embedding vector dimension D in encoder layers.
            encoder_pwff_dimension: point-wise feed forward internal dimension
            encoder_dropout_ratio: dropout ratio to use in encoder layers
            encoder_layers: number of encoder layers in the Encoder.
            decoder_vocabulary_size: max number of tokens in the decoder tokenizer
            decoder_max_time_steps: max sequence length T of decoder layers
            decoder_model_dimension: embedding vector dimension D in decoder layers.
            decoder_pwff_dimension: point-wise feed forward internal dimension
            decoder_dropout_ratio: dropout ratio to use in decoder layers
            decoder_layers: number of decoder layers in the Decoder.
        """
        super().__init__()

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ================================================================================
        # Encoder
        # ================================================================================
        # --------------------------------------------------------------------------------
        # Token embeddings
        # --------------------------------------------------------------------------------
        self.input_embedding: InputEmbedding = InputEmbedding(
            d_model=encoder_model_dimension,
            vocabulary_size=encoder_vocabulary_size,
            dtype=data_type
        )

        # --------------------------------------------------------------------------------
        # Position encoded vectors
        # Citation:
        # --------------------------------------------------------------------------------
        self.encoder_positional_encoding: PositionalEncoding = PositionalEncoding(
            d_model=encoder_model_dimension,
            max_time_steps=encoder_max_time_steps,
        )
        # --------------------------------------------------------------------------------
        # Dropout for the sums of the embeddings and the positional encodings
        # 5.4 Regularization
        # ...
        # In addition, we apply dropout to the sums of the embeddings and the positional
        # encodings in both the encoder and decoder stacks. For the base model, we use a
        # rate of Pdrop = 0.1.
        #
        # Why apply Dropout to Position Encoded tokens removing 10% of tokens in the sequence?
        # https://datascience.stackexchange.com/q/128328/68313
        #
        # TODO: Need to clarify Dropout implementation correctness.
        # Dropout here may remove [CLS], [SEP], [MASK] tokens?
        # https://stackoverflow.com/q/78173751/4281353
        # --------------------------------------------------------------------------------
        self.encoder_dropout: nn.Dropout = nn.Dropout(p=encoder_dropout_ratio)

        self.encoder: nn.Module = Encoder(
            vocabulary_size=encoder_vocabulary_size,
            num_layers=encoder_layers,
            num_heads=NUM_HEADS,
            d_model=encoder_model_dimension,
            dtype=data_type,
            d_ff=encoder_pwff_dimension,
            do_mask=False,
            max_time_steps=encoder_max_time_steps,
            bias=True,
            p_drop=encoder_dropout_ratio
        )

        # ================================================================================
        # Decoder
        # ================================================================================
        # --------------------------------------------------------------------------------
        # Token embeddings
        # The name output_embedding refers to the "output side" of the Transformer model
        # as per the original paper. The input to the decoder is the target sequence
        # shifted one position to the right by inserting the <START> token at the top.
        # --------------------------------------------------------------------------------
        # self.embedding: nn.Embedding = nn.Embedding(
        #     num_embeddings=vocabulary_size,
        #     embedding_dim=d_model
        # )
        # initialize_weights(module=self.embedding)
        self.output_embedding: InputEmbedding = InputEmbedding(
            d_model=decoder_model_dimension,
            vocabulary_size=decoder_vocabulary_size,
            dtype=data_type
        )

        # --------------------------------------------------------------------------------
        # Position encoded vectors
        # --------------------------------------------------------------------------------
        self.decoder_positional_encoding: PositionalEncoding = PositionalEncoding(
            d_model=decoder_model_dimension,
            max_time_steps=decoder_max_time_steps,
        )

        # --------------------------------------------------------------------------------
        # Dropout for the sums of the embeddings and the positional encodings
        # 5.4 Regularization
        # ...
        # In addition, we apply dropout to the sums of the embeddings and the positional
        # encodings in both the encoder and decoder stacks. For the base model, we use a
        # rate of Pdrop = 0.1.
        #
        # Why apply Dropout to Position Encoded tokens removing 10% of tokens in the sequence?
        # https://datascience.stackexchange.com/q/128328/68313
        #
        # TODO: Need to clarify Dropout implementation correctness.
        # Dropout here may remove [CLS], [SEP], [MASK] tokens?
        # https://stackoverflow.com/q/78173751/4281353
        # --------------------------------------------------------------------------------
        self.decoder_dropout: nn.Dropout = nn.Dropout(p=decoder_dropout_ratio)

        self.decoder: nn.Module = Decoder(
            vocabulary_size=decoder_vocabulary_size,
            num_layers=decoder_layers,
            num_heads=NUM_HEADS,
            d_model=decoder_model_dimension,
            dtype=data_type,
            d_ff=decoder_pwff_dimension,
            max_time_steps=decoder_max_time_steps,
            bias=True,
            p_drop=decoder_dropout_ratio
        )

        self.projection: nn.Module = Projection(
            d_model=decoder_model_dimension,
            # num_classes=NUM_CLASSES,
            num_classes=decoder_vocabulary_size,
            dtype=data_type,
            bias=True
        )

        # --------------------------------------------------------------------------------
        # Weight tying: share weights between decoder input embedding and output projection.
        # > 3.4 Embeddings and Softmax:
        # > we also share the same weight matrix between the two embedding layers
        # > and the pre-softmax linear transformation.
        # nn.Embedding.weight shape: (vocab_size, d_model)
        # nn.Linear.weight shape:    (vocab_size, d_model)
        # --------------------------------------------------------------------------------
        self.projection.projection.weight = self.output_embedding.embedding.weight

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

    def _device(self) -> torch.device:
        """Return the device where the model parameters or buffers live.

        Falls back to CPU if no parameters/buffers are present.
        """
        if hasattr(self, "device") and isinstance(self.device, torch.device):
            return self.device
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")

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
            raise ValueError(f"expected x.shape (B, T), got {x.shape}.")
        if y.ndim != 2:
            raise ValueError(f"expected y.shape (B, T), got {y.shape}.")
        if x.device != y.device:
            raise RuntimeError(f"x is on device {x.device} but y is on {y.device}")

        # --------------------------------------------------------------------------------
        # Input Embeddings multiplied by sqrt(d_model).
        # --------------------------------------------------------------------------------
        x = self.input_embedding(indices=x)

        # --------------------------------------------------------------------------------
        # Positional Encoding followed by dropout.
        # > 3.4 Embeddings and Softmax
        # > In addition, we apply dropout to the sums of the embeddings and the
        # > positional encodings in both the encoder and decoder stacks.
        # DO NOT use += as it is in-place operation that can cause back-prop issue.
        # https://stackoverflow.com/a/68600205/4281353
        # https://crazyoscarchang.github.io/2018/10/04/in-pytorch-not-the-same/
        # --------------------------------------------------------------------------------
        x = self.encoder_dropout(x + self.encoder_positional_encoding(x))

        memory: Tensor = self.encoder(x=x)

        # --------------------------------------------------------------------------------
        # Input Embeddings multiplied by sqrt(d_model).
        # --------------------------------------------------------------------------------
        y = self.output_embedding(indices=y)

        # --------------------------------------------------------------------------------
        # Positional Encoding followed by dropout.
        # > 3.4 Embeddings and Softmax
        # > In addition, we apply dropout to the sums of the embeddings and the
        # > positional encodings in both the encoder and decoder stacks.
        # DO NOT use += as it is in-place operation that can cause back-prop issue.
        # https://stackoverflow.com/a/68600205/4281353
        # https://crazyoscarchang.github.io/2018/10/04/in-pytorch-not-the-same/
        # --------------------------------------------------------------------------------
        y = self.decoder_dropout(y + self.decoder_positional_encoding(y))

        # Projection returns log-probabilities. Do NOT pass the Projection output to
        # CrossEntropyLoss as it expects logits and internally applies log-softmax + NLLLoss.
        log_probabilities: Tensor = self.projection(y=self.decoder(y=y, memory=memory))
        return log_probabilities

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

        log_probabilities = self.forward(x=x, y=y)
        predictions = torch.argmax(log_probabilities, dim=-1)
        return predictions

    @torch.no_grad()
    def generate(
            self,
            x: Tensor,
            start_token: int,
            end_token: int,
            max_length: int = DECODER_MAX_TIME_STEPS
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

        _B = x.shape[0]     # pylint: disable=invalid-name

        # Encode source sequence once
        x = self.input_embedding(indices=x)
        x = self.encoder_dropout(x + self.encoder_positional_encoding(x))
        memory = self.encoder(x=x)

        # Start with <START> token for each sequence in batch
        y = torch.full((_B, 1), start_token, dtype=torch.long, device=x.device)

        for _ in range(max_length - 1):
            # Get prediction for the next token
            y_emb = self.output_embedding(indices=y)
            y_emb = self.decoder_dropout(y_emb + self.decoder_positional_encoding(y_emb))
            log_probabilities = self.projection(y=self.decoder(y=y_emb, memory=memory))
            next_token = torch.argmax(log_probabilities[:, -1, :], dim=-1, keepdim=True)

            # Append predicted token to sequence
            y = torch.cat([y, next_token], dim=1)

            # Stop if all sequences have produced end_token
            if (next_token == end_token).all():
                break

        return y
