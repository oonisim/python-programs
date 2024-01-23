"""Module for the Transformers Model
B: Batch size
T: Sequence length or max token size e.g. 512 for BERT. 'T' because of 'Time steps = Sequence length'
D: Dimensions of the model embedding vector, which is d_model in the paper.
H: Number of heads in Multi-head attention
"""
import math
import torch
from torch import (
    Tensor,
    nn
)

from transformer.v1.constant import (
    TYPE_FLOAT,
)
from transformer.v1.utility import (
    softmax
)
from typing import (
    Optional
)


torch.manual_seed(42)


def split(
        x: Tensor,          # pylint: disable=invalid-name
        h: int              # pylint: disable=invalid-name
) -> Tensor:
    """Split an embedding vector into h segments where each segment has d dimensions
    and gets attended by an attention head. Instead of physically split, reshape the
    embedding vector from (B,T,D) into (B,h,T,d) so that each attention head
    attends (T,d).

    Citation:
    > Instead of performing a single attention function with d_model-dimensional
    > keys, values and queries, we found it beneficial to linearly project
    > the queries, keys and values h times with different, learned linear projections
    > to d_k, d_k and d_v dimensions, respectively.

    Args:
        x: embedding of shape (B,T,D)
        h: number of heads to split the embedding into.

    Returns: split embedding of shape (B,h,T,d) where D = h * d
    """
    B, T, D = x.shape       # pylint: disable=invalid-name
    d = D // h
    return x.view(B, T, h, d).transpose(1, 2)   # (B,h,T,d)


def calculate_dot_product_similarities(
        query: Tensor,
        key: Tensor,
) -> Tensor:
    """
    Calculate similarity scores between queries and keys using dot product.

    Args:
        query: embedding vector of query of shape (B, h, T, d_k)
        key: embedding vector of key of shape (B, h, T, d_k)

    Returns: Similarities (closeness) between q and k of shape (B, h, T, T) where
        last (T, T) represents relations between all query elements in T sequence
        against all key elements in T sequence. If T is people in an organization,
        (T,T) represents all (cartesian product) social connections among them.
        The relation considers d_k number of features.
    """
    # --------------------------------------------------------------------------------
    # Relationship between k and q as the first MatMul using dot product similarity:
    # (B, h, T, d_k) @ (B, hH, d_k, T) ---> (B, h, T, T)
    # --------------------------------------------------------------------------------
    similarities = query @ key.transpose(-2, -1)            # dot product
    return similarities                                     # shape:(B, h, T, T)


def scale(
        similarities: Tensor,
        d_k: int
) -> Tensor:
    """
    Standardize the variance of the dot product similarities using the standard deviation
    of the dot product of the normal distributions std=sqrt(d_k) so that the variance will
    be 1.0 approx.

    Citation:
    > While for small values of dk the two mechanisms perform similarly, additive attention
    > outperforms dot product attention without scaling for larger values of dk [3].
    > We suspect that for large values of d_k, the dot products grow large in magnitude,
    > pushing the softmax function into regions where it has extremely small gradients.
    > To counteract this effect, we scale the dot products by sqrt(d_k).

    The last (T, T) of the shape (B,h,T,T) is the matrix that represents the similarities
    as the dot product between (q,k) from every q from sequence length T and k from the
    sequence length T. The dimensions of q and k are both d_k, and q, k are expected to
    follow normal distribution where the mean is 0 and variance is 1. The variance of the
    two normal distributions q, k is expected to be d_k. Hence, standardize the (T,T)
    with its standard deviation std=sqrt(d_k) so that the variance will be approximately 1.
    Then, the later softmax will be smoothed out so that not to pick up higher value.

    Args:
        similarities: Similarities matrix shape (B, h, T, T)
        d_k: dimension of the

    Returns: scaled similarities matrix of shape (B, h, T, T)
    """
    # --------------------------------------------------------------------------------
    # Scaling factor to standardize (div by standard deviation) the product q@k.T
    # of two zero centered normal distributions q, k. The variance of the product
    # is head_size d_k. See https://stats.stackexchange.com/a/52699/105137.
    # --------------------------------------------------------------------------------
    std = torch.sqrt(torch.tensor(d_k, dtype=TYPE_FLOAT))   # standard deviation

    # --------------------------------------------------------------------------------
    # Scale similarities of each head by std so that the variance is approx 1.
    # Scaling regularize the softmax output so as not to overfit to features, by which
    # features in query and key can relate among themselves better.
    # Otherwise, features with higher value will be peaked by softmax, (which is good
    # for use as classification head but not for Bag of Words to incorporate features
    # to make them related), hence only specific features in query and key will be
    # connected.
    # --------------------------------------------------------------------------------
    scaled = similarities / std                             # scaled dot product
    return scaled


def mask(
    similarities: Tensor,
    mask_matrix: Tensor
) -> Tensor:
    """
    Args:
        similarities: matrix to mask of shape (B,H,T,T)
        mask_matrix: boolean matrix of which elements in (T,T) to mask fill.

    Returns: masked similarity matrix
    """
    # --------------------------------------------------------------------------------
    # mask to make uni-direction (left to right only) for algorithm such as GPT.
    # Skip masking for bi-directional e.g .BERT,
    # --------------------------------------------------------------------------------
    # exp(-inf) = 0 masks the similarities so that it will be uni-directional.
    assert (
        similarities.ndim == 4 and                              # (B,H,T,T)
        similarities.shape[-2] == similarities.shape[-1] and
        similarities.shape[-1] == mask_matrix.shape[-1]
    )
    masked = similarities.masked_fill(mask=mask_matrix, value=float('-inf'))
    return masked


def calculate_attentions(
        similarities,
        values
):
    """
    For every q element, create a Bag of Words that encodes the relationships with
    other elements (including itself) in T, using (q,k) relationship value as the
    strength of the relationships.

    Citation:
    > On each of these projected versions of queries, keys and values we then perform
    > the attention function in parallel, yielding d_v-dimensional output values.

    ```
    bows = []
    for row in similarities:                    # similarity matrix of shape (T,T)
        bow = sum([                             # bow:shape(d_v,)
            # each column in row is (q,k) similarity score s
            s*v for (s,v) in zip(row,values)    # k:shape(), v:shape(d_v,)
=        ])
        bows.append(bow)                        # bows:shape(T,d_v)
    ```

    Args:
        similarities: q to k relationship strength matrix of shape (B, h, T, T)
        values: elements of sequence with length T of shape (B, h, T, d_v)

    Returns: Bag of Words for every q element of shape (B, h, T, d_v)
    """
    return similarities @ values     # (B,h,T,T) @ (B,h,T,d_v) -> (B,h,T,d_v)


class ScaledDotProductAttention(nn.Module):
    """
    Class to implement Scaled Dot Product Attention (Figure 2 left in the paper).
    """
    def __init__(self, do_mask: bool, max_time_steps: Optional[int]):
        """
        Args:
            max_time_steps: max sequence length or time steps T
        """
        mask_matrix: Optional[Tensor]
        super().__init__()
        if do_mask:
            mask_matrix = torch.tril(torch.ones(max_time_steps, max_time_steps)) == 0
        else:
            mask_matrix = None

        self.register_buffer("mask_matrix", mask_matrix)
        assert (
            (not do_mask and self.mask_matrix is None) or
            (do_mask and self.mask_matrix.ndim == 2 and self.mask_matrix.shape[-1] == max_time_steps)
        )

    def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
    ):
        """Calculate the scaled dot product attention.
        Args:
            q: query of shape (B,h,T,d)
            k: key of shape (B,h,T,d)
            v: value of shape (B,h,T,d)
        """
        # --------------------------------------------------------------------------------
        # First MatMul in the Scaled Dot Product Attention to calculate the similarities
        # matrix between (q,k) for every (q,k) combinations in Q, K.
        # This is cartesian product matrix of shape (T, T) for every head and batch.
        # The number of features in similarities matrix is B*H*T*T which will be
        # (32 * 8 * 512 * 512) which is 64M. Each feature has 512 / H = 64 dimensions
        # of float32, hence the size is 16G bytes of memory requirement.
        # --------------------------------------------------------------------------------
        similarities: Tensor = calculate_dot_product_similarities(
            query=q,
            key=k,
        )

        # --------------------------------------------------------------------------------
        # Scale (standardize) the dot product similarity matrix with its standard deviation.
        # --------------------------------------------------------------------------------
        d_k = k.size[-1]  # head size
        similarities = scale(similarities=similarities, d_k=d_k)

        # --------------------------------------------------------------------------------
        # Mask if required
        # --------------------------------------------------------------------------------
        if self.mask_matrix is not None:
            similarities = mask(similarities=similarities, mask_matrix=self.mask_matrix)

        # --------------------------------------------------------------------------------
        # Normalize by softmax.
        # --------------------------------------------------------------------------------
        similarities = softmax(similarities, dim=-1)

        # --------------------------------------------------------------------------------
        # Second MatMul to generate attention value for each token in sequence of length T
        # --------------------------------------------------------------------------------
        attentions: Tensor = calculate_attentions(
            similarities=similarities,
            values=v
        )   # shape: (B,H,T,d)

        return attentions


class MultiHeadAttention(nn.Module):
    """
    Class to implement Multi Head Attention (Figure 2 right in the paper).
    Citation:
    > The encoder is composed of a stack of N = 6 identical layers. Each layer has two
    > sub-layers. The first is a multi-head self-attention mechanism, and the second is
    > a simple, position-wise fully connected feed-forward network. ... To facilitate
    > these residual connections, all sub-layers in the model, as well as the embedding
    > layers, produce outputs of dimension d_model = 512.

    > Instead of performing a single attention function with d_model dimensional
    > keys, values and queries, we found it beneficial to linearly project
    > the queries, keys and values h times with different, learned linear projections
    > to d_k, d_k and d_v dimensions, respectively.
    > On each of these projected versions of queries, keys and values we then perform
    > the attention function in parallel, yielding d_v dimensional output values.
    > In this work we employ h = 8 parallel attention layers, or heads. For each of
    > these we use dk = dv = dmodel /h = 64.

    > We apply dropout to the output of each sub-layer, before it is added to the
    > sub-layer input and normalized. In addition, we apply dropout to the sums of the
    > embeddings and the positional encodings in both the encoder and decoder stacks.
    > For the base model, we use a rate p_drop = 0.1.
    """
    @property
    def D(self) -> int:     # pylint: disable=invalid-name
        """Dimensions (number of features) of the embedding vector d_model.
        """
        return self._D

    @property
    def H(self) -> int:     # pylint: disable=invalid-name
        """Number of attention heads"""
        return self._H

    @property
    def T(self) -> int:     # pylint: disable=invalid-name
        """Max time steps or max sequence length"""
        return self._T

    def __init__(
            self,
            num_heads: int = 8,
            d_model: int = 512,
            dtype: type = TYPE_FLOAT,
            do_mask: bool = False,
            max_time_steps: int = 512,
            bias: bool = True,
            p_drop: float = 0.1,
    ):
        """Multi Head Attention initialization.
        Args:
            num_heads: number of attention heads
            d_model: dimensions of the model embedding vector.
            dtype: data type
            do_mask: True when execute masking to not calculate attention with future time steps
            max_time_steps: max sequence length or time steps T.
            bias: Ture if learning the additive bias at the linear layer.
            p_drop: dropout rate.
        """
        super().__init__()
        self._D: int = d_model              # pylint: disable=invalid-name
        self._H: int = num_heads            # pylint: disable=invalid-name
        self._T: int = max_time_steps       # pylint: disable=invalid-name

        # To transfer embedded token of dim_input features to Q space of d_model features
        self.Wq: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=d_model,
            out_features=d_model,
            dtype=dtype,
            bias=bias
        )
        # To transfer embedded token of dim_input features to K space of d_model features
        self.Wk: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=d_model,
            out_features=d_model,
            dtype=dtype,
            bias=bias
        )
        # To transfer embedded token of dim_input features to V space of d_model features
        self.Wv: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=d_model,
            out_features=d_model,
            dtype=dtype,
            bias=bias
        )
        # Project to apply to the concatenated output of Self Dot Product Attention
        self.Wo: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=d_model,
            out_features=d_model,
            dtype=dtype,
            bias=bias
        )
        self.scaled_dot_product_attention: nn.Module = ScaledDotProductAttention(
            do_mask=do_mask,
            max_time_steps=max_time_steps
        )
        self.dropout: nn.Module = nn.Dropout(
            p=p_drop
        )

    def forward(
            self,
            x
    ):
        """Run multi head attention
        Args:
            x: input embedding vector of shape (B,T,D)

        Returns: Attention values of shape (B,T,D)
        """
        # pylint: disable=invalid-name
        assert x.ndim == 3
        B, T, _D = x.shape      # Batch, Tokens (or sequence length), Dimension
        assert _D == self.D, \
            f"input vector dimension is invalid, expected [{self.D}], got [{_D}]."

        # --------------------------------------------------------------------------------
        # Transfer x into Q, K, V spaces. Corresponds to the first 'Linear' layers in
        # the figure 2 of the original paper. In the paper, there are H number of Linear
        # layers Q, K, V respectively, but no need to physically split into H number of
        # Linear layers. Instead, use one Linear layer Wq, Wk, Wv for Q, K, V respectively.
        # --------------------------------------------------------------------------------
        q: Tensor = self.Wq(x)   # Transfer to Q space. Shape=(B, T, D)
        k: Tensor = self.Wk(x)   # Transfer to K space. Shape=(B, T, D)
        v: Tensor = self.Wv(x)   # Transfer to V space. Shape=(B, T, D)
        assert q.shape == (self.B, self.T, self.D)
        assert k.shape == (self.B, self.T, self.D)

        # --------------------------------------------------------------------------------
        # Split into H segments for multiple heads to attend.
        # --------------------------------------------------------------------------------
        q = split(x=q, h=self.H)    # (B, H, T, d)
        k = split(x=k, h=self.H)    # (B, H, T, d)
        v = split(x=v, h=self.H)    # (B, H, T, d)

        # --------------------------------------------------------------------------------
        # Calculate self attention values
        # --------------------------------------------------------------------------------
        attentions: Tensor = self.scaled_dot_product_attention(q=q, k=k, v=v)
        assert attentions.shape == (B, self.H, T, self.D/self.H)

        # --------------------------------------------------------------------------------
        # Concatenate outputs from heads into the model output with reshape.
        # First (B,H,T,d)->(B,T,H,d) then to (B,T,D)
        # --------------------------------------------------------------------------------
        attentions = attentions.transpose(2, 1).view(B, T, -1)
        assert attentions.shape == (B, T, self.D)

        # --------------------------------------------------------------------------------
        # Last Wo Linear projection
        # --------------------------------------------------------------------------------
        attentions = self.Wo(attentions)    # (B,T,D)@(D,D) -> (B,T,D)

        # --------------------------------------------------------------------------------
        # Dropout
        # --------------------------------------------------------------------------------
        attentions = self.dropout(attentions)
        assert attentions.shape == (B, T, self.D)

        return attentions.contiguous()


class PositionwiseFeedForward(nn.Module):
    """Class to implementation of Position-wise Feed-Forward Networks.
    This is a single hidden layer nural network with ReLU activation.

    Citation:
    > The encoder is composed of a stack of N = 6 identical layers. Each layer has two
    > sub-layers. The first is a multi-head self-attention mechanism, and the second is
    > a simple, position-wise fully connected feed-forward network. ... To facilitate
    > these residual connections, all sub-layers in the model, as well as the embedding
    > layers, produce outputs of dimension d_model = 512.

    > We apply dropout to the output of each sub-layer, before it is added to the
    > sub-layer input and normalized. In addition, we apply dropout to the sums of the
    > embeddings and the positional encodings in both the encoder and decoder stacks.
    > For the base model, we use a rate p_drop = 0.1.

    > each of the layers in our encoder and decoder contains a fully connected feed-forward
    > network, which is applied to each position separately and identically. This consists
    > of two linear transformations with a ReLU activation in between.
    > Another way of describing this is as two convolutions with kernel size 1.
    > The dimensionality of input and output is dmodel = 512, and the inner-layer has
    > dimensionality of d_ff = 2048.
    """
    def __init__(
            self,
            d_model: int = 512,
            d_ff: int = 2048,
            dtype: type = TYPE_FLOAT,
            bias: bool = True,
            p_drop: float = 0.1
    ):
        """Initialize the class
        Args:
            d_model: dimensions of the model embedding vector.
            d_ff: dimensions of the hidden layer output vector
            dtype: data type
            bias: True to learn additive bias in the layer.
            p_drop: dropout rate.
        """
        super().__init__()
        self.W1: nn.Module = nn.Linear(
            in_features=d_model, out_features=d_ff, bias=bias, dtype=dtype
        )
        self.relu = nn.ReLU()
        self.W2: nn.Module = nn.Linear(
            in_features=d_ff, out_features=d_model, bias=bias, dtype=dtype
        )
        self.dropout: nn.Module = nn.Dropout(p=p_drop)

    def forward(self, x):
        """Feed-forward neural network forward propagation
        Args:
            x: input embedding vector of shape (B,T,D)

        Returns: output embedding vector of shape (B,T,D)
        """
        return self.dropout(self.W2(self.relu(self.W1(x))))


class EncodeLayer(nn.Module):
    """Class to implement Encoder Layer"""
    def __init__(
            self,
            num_heads: int = 8,
            d_model: int = 512,
            dtype: type = TYPE_FLOAT,
            d_ff: int = 2048,
            do_mask: bool = False,
            max_time_steps: int = 512,
            bias: bool = True,
            p_drop: float = 0.1,
            eps: float = 1e-5
    ):
        """
        Args:
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
        self.multihead_attention: MultiHeadAttention = MultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model,
            dtype=dtype,
            do_mask=do_mask,
            max_time_steps=max_time_steps,
            bias=bias,
            p_drop=p_drop
        )
        self.layernorm_multihead: nn.LayerNorm = nn.LayerNorm(
            normalized_shape=d_model,
            eps=eps,
            dtype=dtype
        )
        self.positionwise_feedforward: PositionwiseFeedForward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dtype=dtype,
            bias=bias,
            p_drop=p_drop
        )
        self.layernorm_positionwise: nn.LayerNorm = nn.LayerNorm(
            normalized_shape=d_model,
            eps=eps,
            dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """Embedding
        Args:
            x: embedding vector of shape (B,T,D)
        """
        x += self.layernorm_multihead(self.multihead_attention(x))
        x += self.layernorm_positionwise(self.positionwise_feedforward(x))
        return x


class PositionalEncoding(nn.Module):
    """Class to implement the positional encoding.
    Taken from https://nlp.seas.harvard.edu/annotated-transformer/

    Citation:
    > In addition, we apply dropout to the sums of the embeddings and
    > the positional encodings in both the encoder and decoder stacks.
    > For the base model, we use a rate p_drop = 0.1.
    """
    def __init__(
            self,
            d_model,
            max_positional_length: int = 5000,
            p_drop: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)

        # Compute the positional encodings once in log space.
        position_encoding = torch.zeros(max_positional_length, d_model)
        position = torch.arange(0, max_positional_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer("position_encoding", position_encoding)

    def forward(self, x: Tensor):
        """Positional encoding
        Args:
            x: embedding vector of shape (B,T,D)
        """
        x = x + self.position_encoding[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Encoder(nn.Module):
    """Class to implement Transformer Encoder.
    Citation:
    > In addition, we apply dropout to the sums of the embeddings and
    > the positional encodings in both the encoder and decoder stacks.
    > For the base model, we use a rate p_drop = 0.1.
    """
    @property
    def D(self) -> int:
        """Dimension of the model embedding vector
        """
        return self._D

    def __init__(
            self,
            vocabulary_size: int,
            max_positional_length: int = 5000,
            num_layers: int = 6,
            num_heads: int = 8,
            d_model: int = 512,
            dtype: type = TYPE_FLOAT,
            d_ff: int = 2048,
            do_mask: bool = False,
            max_time_steps: int = 512,
            bias: bool = True,
            p_drop: float = 0.1,
            eps: float = 1e-5
    ):
        super().__init__()
        self._D: int = d_model

        self.embedding: nn.Embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=d_model
        )
        self.positional_encoding: PositionalEncoding = PositionalEncoding(
            d_model=d_model,
            max_positional_length=max_positional_length,
            p_drop=p_drop
        )
        self.dropout: nn.Dropout = nn.Dropout(p=p_drop)
        self.layers: nn.ModuleList = nn.ModuleList([
            EncodeLayer(
                num_heads=num_heads, d_model=d_model, dtype=dtype, d_ff=d_ff,
                do_mask=do_mask, max_time_steps=max_time_steps, bias=bias,
                p_drop=p_drop, eps=eps
            ) for _ in range(num_layers)
        ])

    def forward(self, indices: Tensor):
        """Encode
        Args:
            indices: indices to tokens
        """
        B, T = indices.shape

        # Embedding.
        # > In the embedding layers, we multiply those weights by dmodel .
        x = self.embedding(indices) * math.sqrt(self.D)

        # Positional embedding.
        positions = torch.arange(0, T, dtype=torch.long, device=indices.device)
        x += self.positional_encoding(positions)
        x = self.dropout(x)

        # Encoder stack
        for _layer in self.layers:
            x = _layer(x)

        return x
