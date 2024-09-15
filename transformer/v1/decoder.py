"""Module of the Decoder stack in the Transformers Model
"""
import torch
from torch import (
    Tensor,
    nn
)

from transformer.v1.constant import (
    TYPE_FLOAT,
    DIM_MODEL,
    NUM_LAYERS,
    NUM_HEADS,
    MAX_SEQUENCE_LENGTH,
)
from transformer.v1.common import (
    PositionalEncoding,
    InputEmbedding,
    MultiHeadAttention,
    PositionwiseFeedForward,
)


class DecodeLayer(nn.Module):
    """Class to implement Decoder Layer
    > We employ a residual connection around each of the two sub-layers, followed
    > by layer normalization. That is, the output of each sub-layer is
    > LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented
    > by the sub-layer itself.

    > Residual Dropout:
    > We apply dropout to the output of each sub-layer, before it is added to the
    > sub-layer input and normalized. In addition, we apply dropout to the sums of
    > the embeddings and the positional encodings in both the encoder and decoder
    > stacks.

    > 3.2.3
    > In "encoder-decoder attention" layers, the queries come from the previous decoder
    > layer, and the memory keys and values come from the output of the encoder.
    > This allows every position in the decoder to attend over all positions in the
    > input sequence. This mimics the typical encoder-decoder attention mechanisms in
    > sequence-to-sequence models.
    >
    > self-attention layers in the decoder allow each position in the decoder to
    > attend to all positions in the decoder up to and including that position.
    > We need to prevent leftward information flow in the decoder to preserve the
    > auto-regressive property
    > [NOTE]:
    > As in Figure 2 of the paper, the masked attention is first attention only.
    """
    def __init__(
            self,
            i_layer: int,
            num_heads: int = NUM_HEADS,
            d_model: int = DIM_MODEL,
            dtype: Tensor.dtype = TYPE_FLOAT,
            d_ff: int = 2048,
            max_time_steps: int = MAX_SEQUENCE_LENGTH,
            bias: bool = True,
            p_drop: float = 0.1,
            eps: float = 1e-5
    ):
        """
        Args:
            i_layer: layer index (i-th layer from the bottom)
            num_heads: number of attention heads
            d_model: dimensions of the model embedding vector.
            d_ff: dimensions of the positionwise feed forward hidden layer output vector
            max_time_steps: max sequence length or time steps T.
            bias: Ture if learning the additive bias at the linear layer.
            p_drop: dropout rate.
            eps: epsilon of LayerNorm
        """
        super().__init__()

        # --------------------------------------------------------------------------------
        # Causal masked attention
        # --------------------------------------------------------------------------------
        self.layer_norm_input: nn.LayerNorm = nn.LayerNorm(     # Normalize decoder input
            normalized_shape=d_model,
            eps=eps,
            dtype=dtype
        )
        # Attention to historic time sequence, not future
        self.causal_self_attention: MultiHeadAttention = MultiHeadAttention(
            i_layer=i_layer,
            num_heads=num_heads,
            d_model=d_model,
            dtype=dtype,
            do_mask=True,
            max_time_steps=max_time_steps,
            bias=bias,
        )
        self.dropout_causal: nn.Module = nn.Dropout(p=p_drop)

        # --------------------------------------------------------------------------------
        # Cross attention
        # Layer normalization on both from encoder memory and from decoder causal attention.
        # --------------------------------------------------------------------------------
        self.layer_norm_memory: nn.LayerNorm = nn.LayerNorm(    # Normalize encoder memory
            normalized_shape=d_model,
            eps=eps,
            dtype=dtype
        )
        self.layer_norm_causal: nn.LayerNorm = nn.LayerNorm(    # Normalize causal attention
            normalized_shape=d_model,
            eps=eps,
            dtype=dtype
        )
        self.cross_attention: MultiHeadAttention = MultiHeadAttention(
            i_layer=i_layer,
            num_heads=num_heads,
            d_model=d_model,
            dtype=dtype,
            do_mask=False,
            max_time_steps=max_time_steps,
            bias=bias,
        )
        self.dropout_cross: nn.Module = nn.Dropout(p=p_drop)

        # --------------------------------------------------------------------------------
        # Point-wise Feed Forward
        # --------------------------------------------------------------------------------
        self.layer_norm_cross: nn.LayerNorm = nn.LayerNorm(     # Normalize cross attention
            normalized_shape=d_model,
            eps=eps,
            elementwise_affine=True,
            bias=True,
            dtype=dtype
        )
        self.feedforward: PositionwiseFeedForward = PositionwiseFeedForward(
            i_layer=i_layer,
            d_model=d_model,
            d_ff=d_ff,
            dtype=dtype,
            bias=bias,
        )
        self.dropout_feedforward: nn.Module = nn.Dropout(p=p_drop)

    def forward(
            self,
            x: Tensor,
            memory: Tensor
    ) -> Tensor:
        """Decode.
        Args:
            x: embedding vector from the decoder layer of shape (B,T,D)
            memory: encoder embedding vector of shape (B, T, D)
        """
        # > 3.2.3
        # > In "encoder-decoder attention" layers, the queries come from the previous decoder
        # > layer, and the memory keys and values come from the output of the encoder.
        # > This allows every position in the decoder to attend over all positions in the
        # > input sequence. This mimics the typical encoder-decoder attention mechanisms in
        # > sequence-to-sequence models.
        #
        # Token q from the target sequence identifies the relationship strengths
        # with each token v in the source sequence. This mimics the original attention
        # mechanism (cursor) in the sequence to sequence model.
        _q = self.layer_norm_input(x)
        _k = _v = self.layer_norm_memory(memory)
        x = x + self.dropout_causal(self.causal_self_attention(q=_q, k=_k, v=_v))
        x = x + self.dropout_cross(self.cross_attention(self.layer_norm_causal(x)))
        x = x + self.dropout_feedforward(self.feedforward(self.layer_norm_cross(x)))
        assert torch.all(torch.isfinite(x))
        return x


class Decoder(nn.Module):
    """Class to implement Transformer Encoder.
    Citation:
    > In addition, we apply dropout to the sums of the embeddings and
    > the positional encodings in both the encoder and decoder stacks.
    > For the base model, we use a rate p_drop = 0.1.
    """
    @property
    def D(self) -> int:     # pylint: disable=invalid-name
        """Dimension of the model embedding vector
        """
        return self._D

    def __init__(
            self,
            vocabulary_size: int,
            num_layers: int = NUM_LAYERS,
            num_heads: int = NUM_HEADS,
            d_model: int = DIM_MODEL,
            dtype: Tensor.dtype = TYPE_FLOAT,
            d_ff: int = 2048,
            max_time_steps: int = MAX_SEQUENCE_LENGTH,
            bias: bool = True,
            p_drop: float = 0.1,
            eps: float = 1e-5
    ):
        super().__init__()
        self._D: int = d_model      # pylint: disable=invalid-name
        assert vocabulary_size > 0, f"invalid vocabulary size [{vocabulary_size}]."

        # --------------------------------------------------------------------------------
        # Token embeddings
        # --------------------------------------------------------------------------------
        # self.embedding: nn.Embedding = nn.Embedding(
        #     num_embeddings=vocabulary_size,
        #     embedding_dim=d_model
        # )
        # initialize_weights(module=self.embedding)
        self.output_embedding: InputEmbedding = InputEmbedding(
            d_model=d_model,
            vocabulary_size=vocabulary_size,
            dtype=dtype
        )

        # --------------------------------------------------------------------------------
        # Position encoded vectors
        # --------------------------------------------------------------------------------
        self.positional_encoding: PositionalEncoding = PositionalEncoding(
            d_model=d_model,
            max_time_steps=max_time_steps,
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
        self.dropout: nn.Dropout = nn.Dropout(p=p_drop)
        # --------------------------------------------------------------------------------
        # Decoder layers
        # --------------------------------------------------------------------------------
        self.layers: nn.ModuleList = nn.ModuleList([
            DecodeLayer(
                i_layer=_layer,
                num_heads=num_heads, d_model=d_model, dtype=dtype, d_ff=d_ff,
                max_time_steps=max_time_steps, bias=bias,
                p_drop=p_drop, eps=eps
            ) for _layer in range(num_layers)
        ])

    def forward(
            self,
            indices: Tensor,
            memory: Tensor,
    ) -> Tensor:
        """Decode the input embeddings.
        Note that the input to the decoder during the training is the target sequence
        shifted one position to the right by inserting the <START> token at the top
        that signals the beginning of the sentence. Hence, there will be T predictions
        for each sequence.

        Args:
            indices: indices to target sequence tokens of shape (B, T)
            memory: encoder embeddings

        Returns: Decoder next token predictions of shape (B, T, D)
        """
        assert torch.is_tensor(indices) and indices.ndim == 2   # shape (B, T)
        _B, _T = indices.shape        # pylint: disable=invalid-name

        # --------------------------------------------------------------------------------
        # Input Embeddings multiplied by sqrt(d_model).
        # --------------------------------------------------------------------------------
        x = self.output_embedding(indices=indices)
        assert x.shape == (_B, _T, self.D)

        # --------------------------------------------------------------------------------
        # Positional Encoding followed by dropout.
        # > 3.4 Embeddings and Softmax
        # > In addition, we apply dropout to the sums of the embeddings and the
        # > positional encodings in both the encoder and decoder stacks.
        # DO NOT use += as it is in-place operation that can cause back-prop issue.
        # https://stackoverflow.com/a/68600205/4281353
        # https://crazyoscarchang.github.io/2018/10/04/in-pytorch-not-the-same/
        # --------------------------------------------------------------------------------
        x = self.dropout(x + self.positional_encoding(x))   # (B,T,D) + (1,T,D)
        assert x.shape == (_B, _T, self.D)
        assert torch.all(torch.isfinite(x))

        # --------------------------------------------------------------------------------
        # N x Encode Layers
        # --------------------------------------------------------------------------------
        for _layer in self.layers:
            x = _layer(x=x, memory=memory)

        assert x.shape == (_B, _T, self.D)
        return x
