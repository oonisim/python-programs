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
from .common import (
    InputEmbedding,
    PositionalEncoding,
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
    # pylint: disable=too-many-arguments
    def __init__(
            self,
            data_type = TYPE_FLOAT,
            encoder_max_time_steps: int = ENCODER_MAX_TIME_STEPS,
            encoder_vocabulary_size: int = ENCODER_MAX_TOKENS,
            encoder_model_dimension: int = DIM_MODEL,
            encoder_pwff_dimension: int = ENCODER_PWFF_DIM,
            encoder_dropout_ratio: float = ENCODER_DROPOUT_RATIO,
            encoder_layers: int = ENCODER_LAYERS,
            encoder_num_heads: int = NUM_HEADS,
            decoder_max_time_steps: int = DECODER_MAX_TIME_STEPS,
            decoder_vocabulary_size: int = DECODER_MAX_TOKENS,
            decoder_model_dimension: int = DIM_MODEL,
            decoder_pwff_dimension: int = DECODER_PWFF_DIM,
            decoder_dropout_ratio: float = DECODER_DROPOUT_RATIO,
            decoder_layers: int = DECODER_LAYERS,
            decoder_num_heads: int = NUM_HEADS,
            decoder_tie_weights: bool = True,
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
            encoder_num_heads: number of attention heads in encoder (must divide encoder_model_dimension)
            decoder_vocabulary_size: max number of tokens in the decoder tokenizer
            decoder_max_time_steps: max sequence length T of decoder layers
            decoder_model_dimension: embedding vector dimension D in decoder layers.
            decoder_pwff_dimension: point-wise feed forward internal dimension
            decoder_dropout_ratio: dropout ratio to use in decoder layers
            decoder_layers: number of decoder layers in the Decoder.
            decoder_num_heads: number of attention heads in decoder (must divide decoder_model_dimension)
            decoder_tie_weights: whether to share weights between input embedding and output projection.
        """
        super().__init__()
        self._cached_device: Optional[torch.device] = None
        self.encoder_max_time_steps: int = encoder_max_time_steps
        self.decoder_max_time_steps: int = decoder_max_time_steps

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
            num_heads=encoder_num_heads,
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

        # --------------------------------------------------------------------------------
        # Decoder layers
        # --------------------------------------------------------------------------------
        self.decoder: nn.Module = Decoder(
            vocabulary_size=decoder_vocabulary_size,
            num_layers=decoder_layers,
            num_heads=decoder_num_heads,
            d_model=decoder_model_dimension,
            dtype=data_type,
            d_ff=decoder_pwff_dimension,
            max_time_steps=decoder_max_time_steps,
            bias=True,
            p_drop=decoder_dropout_ratio
        )

        # --------------------------------------------------------------------------------
        # Final Layer Norm before the projection for stability.
        # Softmax is an amplifier that makes large logits larger and
        # small logits smaller.
        #
        # A large input to softmax can cause gradient spikes or numerical
        # instability (inf/Nan) because of exp/log operations.
        # A small input can cause tiny gradients and slow learning.
        #
        # By applying a final layer normalization before the projection,
        # the distribution of per-vector is well-behaved by centering it
        # (mean ≈ 0) and scaling it (std ≈ 1). Then Softmax sees logits that
        # are not arbitrarily large or tiny.
        #
        # However, PreLN based transformers already manages the distribution
        # of the signal well, so the final layer norm is not strictly necessary.
        # A final LN often can give small, consistent stability gains and
        # sometimes a small metric bump, but it’s not a make‑or‑break component.
        # Many modern models include it as it is not costly, but some models omit it.
        # --------------------------------------------------------------------------------
        self.final_decoder_norm = nn.LayerNorm(
            # LayerNorm normalizes over the last dimension of tensor
            # when normalized_shape is an int, then normalized_shape
            # must specify the expected size of the last dimension D.
            normalized_shape=decoder_model_dimension,
            dtype=data_type
        )

        # --------------------------------------------------------------------------------
        # LM projection head to map decoder output to vocabulary log probabilities.
        # Beware Projection outputs log-probabilities by default with log-softmax.
        # Do NOT pass the Projection output to CrossEntropyLoss unless return-logits=True.
        # --------------------------------------------------------------------------------
        # Weight tying: share weights between decoder input embedding and output projection.
        # > 3.4 Embeddings and Softmax:
        # > we also share the same weight matrix between the two embedding layers
        # > and the pre-softmax linear transformation.
        # nn.Embedding.weight shape: (vocab_size, d_model)
        # nn.Linear.weight shape:    (vocab_size, d_model)
        #
        # When tying, Projection(bias=False) to assures symmetry with Embedding that has
        # no bias by design. Keeping bias on the projection would break the symmetry that
        # tying is meant to enforce (embedding is a pure row lookup into W, so the
        # projection should be y @ W^T without an additive bias term).
        #
        # Embedding weight E ∈ R^(V × D) meaning shape:(V, D) where V is vocabulary size.
        # Projection weight W ∈ R^(D × V). Hence, W = Eᵀ.

        # Then if bias=False, Embedding space alone determines Prediction.
        # Embedding(index) = E[index] -> token embedding vector d.
        # Projection(bias=False)(d) = softmax(d@Eᵀ) --argmax--> index.
        # This gives symmetry (index -> d at Embedding) and (d --> index at Projection).
        #
        # Projection(bias=True)(d) = softmax(d@Eᵀ + b) breaks symmetry by the bias b.
        # Now Prediction depends on something outside embedding space.
        # Therefore, No more "Embedding space alone determines Prediction".
        #
        # Mathematical elegance of weight-tying is treating the embedding and the projection
        # as inverse operations of each other, which is symmetry.
        # --------------------------------------------------------------------------------
        self.projection: nn.Module = Projection(
            d_model=decoder_model_dimension,
            # num_classes=NUM_CLASSES,
            num_classes=decoder_vocabulary_size,
            dtype=data_type,
            bias=False if decoder_tie_weights else True # pylint: disable=simplifiable-if-expression
        )

        if decoder_tie_weights:
            assert (
                self.projection.projection.weight.shape ==
                self.output_embedding.embedding.weight.shape
            ), (
                "Shape mismatch for weight tying: projection "
                f"{self.projection.projection.weight.shape} "
                f"!= embedding {self.output_embedding.embedding.weight.shape}"
            )
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
        """Return the single device where all model parameters and buffers live.

        The result is cached and invalidated when the model is moved via
        .to() / .cuda() / .cpu() (all of which funnel through _apply).

        Raises:
            RuntimeError: If parameters/buffers are spread across multiple devices.

        Falls back to CPU if no parameters/buffers are present.
        """
        if self._cached_device is not None:
            return self._cached_device

        devices = {p.device for p in self.parameters()}
        devices |= {b.device for b in self.buffers()}
        if len(devices) > 1:
            raise RuntimeError(
                f"Model parameters/buffers are on multiple devices: {devices}. "
                "Move the entire model to a single device with model.to(device)."
            )

        # next(devices) can be cpu, cuda:0, cuda:1, mps
        self._cached_device = torch.device("cpu") if len(devices) == 0 else next(iter(devices))
        return self._cached_device

    def _apply(self, fn):
        """Override to invalidate cached device when model is moved.
        _apply(self, fn) hooks .to(), .cuda(), .cpu(), .float(), .half().
        It recursively walks every parameter and buffer to apply fn to each.
        This is where device/dtype changes actually happen.
        Whereas, apply(self, fn) is used for custom weight initialization
        (model.apply(init_weights)). It has nothing to do with device moves.
        """
        self._cached_device = None
        return super()._apply(fn)

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
            y: Tensor,
            source_pad_mask: Optional[Tensor] = None,
            target_pad_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Run Transformer encoder/decoder
        DO NOT place non-differentiable functions e.g, argmax in forward.
        This returns log-probabilities. Do NOT pass the Projection output to CrossEntropyLoss
        as it expects logits and internally applies log-softmax + NLLLoss.

        Args:
            x: Encoder source sequences as token indices of shape (B, Te).
            y: Decoder target sequences as token indices of shape (B, Td).
            source_pad_mask: optional padding mask of shape (B, Te) for encoder
                sequence, where True indicates padding positions to be masked out
                in attention.
            target_pad_mask: optional padding mask of shape (B, Td) for decoder
                sequence, where True indicates padding positions to be masked out
                in causal self-attention.

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

        model_device = self._device()
        if x.device != model_device:
            raise RuntimeError(
                f"Input x is on {x.device} but model is on {model_device}. "
                "Move your inputs explicitly with tensor.to(device)."
            )
        if y.device != model_device:
            raise RuntimeError(
                f"Input y is on {y.device} but model is on {model_device}. "
                "Move your inputs explicitly with tensor.to(device)."
            )

        if x.shape[1] > self.encoder_max_time_steps:
            raise ValueError(
                f"Encoder input sequence length {x.shape[1]} exceeds "
                f"model max_time_steps {self.encoder_max_time_steps}; "
                "consider increasing max_time_steps or truncating inputs."
            )
        if y.shape[1] > self.decoder_max_time_steps:
            raise ValueError(
                f"Decoder input sequence length {y.shape[1]} exceeds "
                f"model max_time_steps {self.decoder_max_time_steps}; "
                "consider increasing max_time_steps or truncating inputs."
            )

        # --------------------------------------------------------------------------------
        # Input Embeddings multiplied by sqrt(d_model).
        # --------------------------------------------------------------------------------
        x: Tensor = self.input_embedding(indices=x)

        # --------------------------------------------------------------------------------
        # Positional Encoding followed by dropout.
        # > 3.4 Embeddings and Softmax
        # > In addition, we apply dropout to the sums of the embeddings and the
        # > positional encodings in both the encoder and decoder stacks.
        # DO NOT use += as it is in-place operation that can cause back-prop issue.
        # https://stackoverflow.com/a/68600205/4281353
        # https://crazyoscarchang.github.io/2018/10/04/in-pytorch-not-the-same/
        # --------------------------------------------------------------------------------
        x: Tensor = self.encoder_dropout(x + self.encoder_positional_encoding(x))

        memory: Tensor = self.encoder(x=x, source_pad_mask=source_pad_mask)

        # --------------------------------------------------------------------------------
        # Output/Decoder Embeddings multiplied by sqrt(d_model).
        # --------------------------------------------------------------------------------
        y: Tensor = self.output_embedding(indices=y)

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

        # --------------------------------------------------------------------------------
        # Decoder layers with cross attention to encoder output (memory).
        # Pass source_pad_mask to prevent attending to padded encoder positions.
        # Pass target_pad_mask to prevent attending to padded decoder positions.
        # --------------------------------------------------------------------------------
        y = self.decoder(
            y=y,
            memory=memory,
            source_pad_mask=source_pad_mask,
            target_pad_mask=target_pad_mask
        )

        # --------------------------------------------------------------------------------
        # Projection to vocabulary log probabilities.
        # Projection returns log-probabilities. Do NOT pass the Projection output to
        # CrossEntropyLoss as it expects logits and internally applies log-softmax + NLLLoss.
        # --------------------------------------------------------------------------------
        log_probabilities: Tensor = self.projection(y=self.final_decoder_norm(y))
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

        model_device = self._device()
        if x.device != model_device:
            raise RuntimeError(
                f"Input x is on {x.device} but model is on {model_device}. "
                "Move your inputs explicitly with tensor.to(device)."
            )

        # Bug fix: Preserve training mode state across evaluation
        # Without this, if evaluate() is called during training, the model stays in eval mode
        # after evaluation completes, causing dropout to remain disabled for subsequent training.
        was_training = self.training
        self.eval()

        try:
            # Use self() instead of self.forward() to properly invoke __call__
            log_probabilities = self(x=x, y=y)
            predictions = torch.argmax(log_probabilities, dim=-1)
            return predictions

        finally:
            # Restore training mode if model was in training before evaluation
            if was_training:
                self.train()

    @torch.no_grad()
    def generate(
            self,
            x: Tensor,
            start_token: int,
            end_token: int,
            max_length: int = DECODER_MAX_TIME_STEPS,
            source_pad_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Autoregressively generate output sequence from source x.

        Args:
            x: Source sequence of shape (B, Te)
            start_token: Token index to start generation (e.g., <START>)
            end_token: Token index to stop generation (e.g., <END>)
            max_length: Maximum number of tokens to generate
            source_pad_mask: Optional padding mask for source sequence (B, Te).
                True indicates padding positions that should not be attended to.
                If None, all positions are attended to (no masking).

        Returns: Generated token indices of shape (B, Td) where Td <= max_length
        """
        if x.ndim != 2:
            msg: str = f"expected x.shape (B, T), got {x.shape}."
            logger.error(msg)
            raise ValueError(msg)

        # Validate max_length to prevent positional encoding overflow
        if max_length > self.decoder_max_time_steps:
            raise ValueError(
                f"max_length ({max_length}) exceeds decoder_max_time_steps "
                f"({self.decoder_max_time_steps}). Positional encoding will fail."
            )

        # Warn the user about device/dtype mismatches instead of moving/casting.
        if x.device != self._device():
            raise RuntimeError(
                f"Input tensor x is on {x.device} but model on {self._device()}. "
                "Move your inputs explicitly to the desired device."
            )

        if x.shape[1] > self.encoder_max_time_steps:
            raise ValueError(
                f"Encoder input sequence length {x.shape[1]} exceeds "
                f"model max_time_steps {self.encoder_max_time_steps}; "
                "consider increasing max_time_steps or truncating inputs."
            )

        _B = x.shape[0]     # pylint: disable=invalid-name

        # Bug fix: Switch to eval mode to disable dropout for consistent generation
        # Without this, dropout remains active if called during training, producing noisy outputs
        # Save training state and restore after generation to avoid side effects
        was_training = self.training
        self.eval()

        try:
            # Encode source sequence once
            x = self.input_embedding(indices=x)
            x = self.encoder_dropout(x + self.encoder_positional_encoding(x))
            memory = self.encoder(x=x, source_pad_mask=source_pad_mask)

            # Start with <START> token for each sequence in batch
            y = torch.full((_B, 1), start_token, dtype=torch.long, device=x.device)

            for _ in range(max_length - 1):
                # Get prediction for the next token
                y_emb = self.output_embedding(indices=y)
                y_emb = self.decoder_dropout(y_emb + self.decoder_positional_encoding(y_emb))

                # Apply decoder with source padding mask for cross-attention
                decoder_output = self.decoder(
                    y=y_emb,
                    memory=memory,
                    source_pad_mask=source_pad_mask
                )
                log_probabilities = self.projection(y=self.final_decoder_norm(decoder_output))

                next_token = torch.argmax(log_probabilities[:, -1, :], dim=-1, keepdim=True)

                # Append predicted token to sequence
                y = torch.cat([y, next_token], dim=1)

                # Stop if all sequences have produced end_token
                if (next_token == end_token).all():
                    break

            return y

        finally:
            # Restore training mode if model was in training before generation
            if was_training:
                self.train()
