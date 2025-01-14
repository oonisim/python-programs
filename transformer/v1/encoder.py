"""Module of the Encoder stack in the Transformers Model
"""
import torch
from torch import (
    Tensor,
    nn
)

from transformer.v1.constant import (
    TYPE_FLOAT,
    DIM_MODEL,
    DIM_PWFF_HIDDEN,
    MAX_TIME_STEPS,
    NUM_LAYERS,
    NUM_HEADS,
    DROPOUT_RATIO,
    EPSILON,
)
from transformer.v1.common import (
    PositionalEncoding,
    InputEmbedding,
    MultiHeadAttention,
    PositionwiseFeedForward,
)


class EncodeLayer(nn.Module):
    """Class to implement Encoder Layer"""
    def __init__(
            self,
            i_layer: int,
            num_heads: int = NUM_HEADS,
            d_model: int = DIM_MODEL,
            dtype: Tensor.dtype = TYPE_FLOAT,
            d_ff: int = DIM_PWFF_HIDDEN,
            do_mask: bool = False,
            max_time_steps: int = MAX_TIME_STEPS,
            bias: bool = True,
            p_drop: float = DROPOUT_RATIO,
            eps: float = EPSILON
    ):
        """
        Args:
            i_layer: layer index (i-th layer from the bottom)
            num_heads: number of attention heads
            d_model: dimensions of the model embedding vector.
            d_ff: dimensions of the positionwise feed forward hidden layer output vector
            do_mask: True when execute masking to not calculate attention with future time steps
            max_time_steps: max sequence length or time steps T.
            bias: Ture if learning the additive bias at the linear layer.
            p_drop: dropout rate.
            eps: epsilon of LayerNorm
        """
        super().__init__()

        self.layer_norm_input: nn.LayerNorm = nn.LayerNorm(             # Normalize encoder input
            normalized_shape=d_model,
            eps=eps,
            dtype=dtype
        )
        self.bidirectional_attention: MultiHeadAttention = MultiHeadAttention(
            i_layer=i_layer,
            num_heads=num_heads,
            d_model=d_model,
            dtype=dtype,
            do_mask=do_mask,
            max_time_steps=max_time_steps,
            bias=bias,
        )
        self.dropout_bidirectional: nn.Module = nn.Dropout(p=p_drop)

        self.layer_norm_bidirectional: nn.LayerNorm = nn.LayerNorm(     # Normalize bidirectional attention
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

    def forward(self, x: Tensor) -> Tensor:
        """Encode.
        Citation:
        > The encoder is composed of a stack of N = 6 identical layers. Each layer has
        > two sub-layers. The first is a multi-head self-attention mechanism, and the
        > second is a simple, positionwise fully connected feed-forward network.

        > We employ a residual connection around each of the two sub-layers, followed
        > by layer normalization. That is, the output of each sub-layer is
        > LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented
        > by the sub-layer itself.

        > Residual Dropout:
        > We apply dropout to the output of each sub-layer, before it is added to the
        > sub-layer input and normalized. In addition, we apply dropout to the sums of
        > the embeddings and the positional encodings in both the encoder and decoder
        > stacks.

        Args:
            x: embedding vector of shape (B,T,D)
        """
        # Update:
        # Original paper applied Layer Normalization after Residual Connection which is
        # called Post Normalization. However, recent approach is Pre-Normalization where
        # LayerNorm is applied before the sub-layer as in https://arxiv.org/pdf/2002.04745.pdf.
        # See https://youtu.be/kCc8FmEb1nY?t=5729 and

        # --------------------------------------------------------------------------------
        # Post Normalization in the original paper. Replaced with Pre Norm as in Update:
        # 1. First sub-layer followed by Dropout before added to sub-layer input x.
        # 2. Add Residual Connection.
        # 3. Layer Normalization.
        # --------------------------------------------------------------------------------
        # x = self.layernorm_multihead(
        #     x + self.dropout_multihead(self.multihead_attention(x))
        # )

        # DO NOT use += as it is in-place operation that can cause back-prop issue.
        # https://stackoverflow.com/a/68600205/4281353
        # https://crazyoscarchang.github.io/2018/10/04/in-pytorch-not-the-same/

        # TODO: Verify _q, _k, _v refer to the same memory location without copy.
        _q = _k = _v = self.layer_norm_input(x)
        x = x + self.dropout_bidirectional(self.bidirectional_attention(q=_q, k=_k, v=_v))

        # --------------------------------------------------------------------------------
        # Post Normalization in the original paper. Replaced with Pre Norm as in Update:
        # 1. Second sub-layer followed by Dropout before added to sub-layer input x.
        # 2. Add Residual Connection.
        # 3. Layer Normalization.
        # --------------------------------------------------------------------------------
        # x = self.layernorm_positionwise(
        #     x + self.dropout_positionwise(self.positionwise_feedforward(x))
        # )
        x = x + self.dropout_feedforward(self.feedforward(self.layer_norm_bidirectional(x)))
        assert torch.all(torch.isfinite(x))
        return x


class Encoder(nn.Module):
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
            d_ff: int = DIM_PWFF_HIDDEN,
            do_mask: bool = False,
            max_time_steps: int = MAX_TIME_STEPS,
            bias: bool = True,
            p_drop: float = DROPOUT_RATIO,
            eps: float = EPSILON
    ):
        super().__init__()
        self._D: int = d_model      # pylint: disable=invalid-name
        assert vocabulary_size > 0, f"invalid vocabulary size [{vocabulary_size}]."

        # --------------------------------------------------------------------------------
        # Token embeddings
        # --------------------------------------------------------------------------------
        self.input_embedding: InputEmbedding = InputEmbedding(
            d_model=d_model,
            vocabulary_size=vocabulary_size,
            dtype=dtype
        )

        # --------------------------------------------------------------------------------
        # Position encoded vectors
        # Citation:
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
        # Encoder layers
        # --------------------------------------------------------------------------------
        self.layers: nn.ModuleList = nn.ModuleList([
            EncodeLayer(
                i_layer=_layer,
                num_heads=num_heads, d_model=d_model, dtype=dtype, d_ff=d_ff,
                do_mask=do_mask, max_time_steps=max_time_steps, bias=bias,
                p_drop=p_drop, eps=eps
            ) for _layer in range(num_layers)
        ])

    def forward(self, x: Tensor):
        """Encode
        Args:
            x: indices to tokens
        """
        assert torch.is_tensor(x) and x.ndim == 2   # shape (B, T)
        _B, _T = x.shape        # pylint: disable=invalid-name

        # --------------------------------------------------------------------------------
        # Input Embeddings multiplied by sqrt(d_model).
        # --------------------------------------------------------------------------------
        x = self.input_embedding(indices=x)
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
            x = _layer(x)

        assert x.shape == (_B, _T, self.D)
        return x
