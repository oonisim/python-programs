"""Module for the Transformers Model"""
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


torch.manual_seed(42)


def split(
        x,                  # pylint: disable=invalid-name
        h: int              # pylint: disable=invalid-name
) -> Tensor:
    """
    Reshape embedding vector from (B,T,M) into (B,h,T,d) so that each head
    of multiple heads attends (T,d).

    Citation:
    > Instead of performing a single attention function with d_model-dimensional
    > keys, values and queries, we found it beneficial to linearly project
    > the queries, keys and values h times with different, learned linear projections
    > to d_k, d_k and d_v dimensions, respectively.

    Args:
        x: embedding of shape (B,T,M)
        h: number of heads to split the embedding into.

    Returns: split embedding of shape (B,h,T,d) where M = h * d
    """
    B, T, M = x.shape       # pylint: disable=invalid-name
    d = M // h
    return x.view(B, T, h, d).transpose(1, 2)   # (B,h,T,d)


def calculate_similarities(
        query,
        key,
        mask=None,
):
    """
    Calculate similarity scores between query and keys using dot product.
    Standardize the variance using sqrt(d_k) so that the variance will be 1.0 approx.

    Args:
        query: embedding vector of query of shape (B, h, T, d_k)
        key: embedding vector of key of shape (B, h, T, d_k)
        mask: mask value to prevent calculating relation with future time step

    Returns: Relationship (closeness) between q and k of shape (B, h, T, T) where
            last (T, T) represents relations between all query elements in T sequence
            against all key elements in T sequence. If T is people in an organization,
            (T,T) represents all (cartesian product) social connections among them.
            The relation considers d_k number of features.
    """
    d_k = key.size[-1]                                      # head size

    # --------------------------------------------------------------------------------
    # Relationship between k and q as the first MatMul using dot product similarity:
    # (B, h, T, d_k) @ (B, hH, d_k, T) ---> (B, h, T, T)
    # --------------------------------------------------------------------------------
    similarities = query @ key.transpose(-2, -1)            # dot product

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
    similarities = similarities / std                       # scaled dot product

    # --------------------------------------------------------------------------------
    # mask to make uni-direction (left to right only) for algorithm such as GPT.
    # Skip masking for bi-directional e.g .BERT,
    # --------------------------------------------------------------------------------
    if mask is not None:
        # TODO: Verify if the logic is correct.
        similarities = similarities.masked_fill(mask == 0, float('-inf'))

    # --------------------------------------------------------------------------------
    # Normalize by softmax.
    # exp(-inf) = 0 masks the similarities so that it will be uni-directional.
    # --------------------------------------------------------------------------------
    similarities = softmax(similarities, dim=-1)

    return similarities                                    # shape:(B, h, T, T)


def calculate_attentions(
        relationships,
        value
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
    for row in relationships:                 # relationship matrix of shape (T,T)
        bow = sum([                           # bow:shape(d_v,)
            k*v for (k,v) in zip(row,value)   # k:shape(), v:shape(d_v,)
        ])
        bows.append(bow)                      # bows:shape(T,d_v)
    ```

    Args:
        relationships: q to k relationship strength matrix of shape (B, h, T, T)
        value: elements of sequence with length T of shape (B, h, T, d_v)

    Returns: Bag of Words for every q element of shape (B, h, T, d_v)
    """
    return relationships @ value     # (B,h,T,T) @ (B,h,T,d_v) -> (B,h,T,d_v)


class MultiHeadAttention(nn.Module):
    """Class to implement Transformer multi head attentions
    Citation:
    > Instead of performing a single attention function with d_model-dimensional
    > keys, values and queries, we found it beneficial to linearly project
    > the queries, keys and values h times with different, learned linear projections
    > to d_k, d_k and d_v dimensions, respectively.
    > On each of these projected versions of queries, keys and values we then perform
    > the attention function in parallel, yielding d_v dimensional output values.

    """
    @property
    def D(self) -> int:     # pylint: disable=invalid-name
        """Input vector dimension"""
        return self._D

    @property
    def M(self) -> int:     # pylint: disable=invalid-name
        """Output attention vector dimension"""
        return self._M

    @property
    def H(self) -> int:     # pylint: disable=invalid-name
        """Number of attention heads"""
        return self._M

    def __init__(
            self,
            num_heads: int,
            dim_input: int,
            dim_output: int,
            bias: bool,
            dropout_ratio: float
    ):
        """Class to implement Multi Head Attention
        Args:
            num_heads: number of attention heads
            dim_input: input embedding vector dimension
            dim_output: output attention vector dimension
            bias: if learn additive bias at the linear layer.
        """
        super().__init__()
        self._H: int = num_heads    # pylint: disable=invalid-name
        self._D: int = dim_input    # pylint: disable=invalid-name
        self._M: int = dim_output   # pylint: disable=invalid-name

        # To transfer embedded token of dim_input features to Q space of dim_output features
        self._Wq: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=dim_input,
            out_features=dim_output,
            dtype=TYPE_FLOAT,
            bias=bias
        )
        # To transfer embedded token of dim_input features to K space of dim_output features
        self._Wk: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=dim_input,
            out_features=dim_output,
            dtype=TYPE_FLOAT,
            bias=bias
        )
        # To transfer embedded token of dim_input features to V space of dim_output features
        self._Wv: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=dim_input,
            out_features=dim_output,
            dtype=TYPE_FLOAT,
            bias=bias
        )

        # Project to apply to the concatenated output of Self Dot Product Attention
        self._Wo: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=dim_output,
            out_features=dim_output,
            dtype=TYPE_FLOAT,
            bias=bias
        )

        self._dropout: nn.Module = nn.Dropout(
            p=dropout_ratio
        )

    def forward(
            self,
            x
    ):
        """Run multi head attention
        Args:
            x: input embedding vector of shape (B,T,D)

        Returns: Attention values of shape (B,T,M)
        """
        # pylint: disable=invalid-name
        B, T, _D = x.shape      # Batch, Tokens (or sequence length), Dimension
        assert _D == self.D, \
            f"input vector dimension is invalid, expected [{self.D}], got [{_D}]."

        q: Tensor = self.Wq(x)   # Transfer to Q space. Shape=(B, T, M)
        k: Tensor = self.Wk(x)   # Transfer to K space. Shape=(B, T, M)
        v: Tensor = self.Wv(x)   # Transfer to V space. Shape=(B, T, M)
        assert q.shape == (self.B, self.T, self.M)
        assert k.shape == (self.B, self.T, self.M)

        # --------------------------------------------------------------------------------
        # Split into multiple heads
        # --------------------------------------------------------------------------------
        q = split(x=q, h=self.H)    # (B, H, T, d_k)
        k = split(x=k, h=self.H)    # (B, H, T, d_k)
        v = split(x=v, h=self.H)    # (B, H, T, d_k)

        # --------------------------------------------------------------------------------
        # Build relationships matrix between (q,k) for (q,k) combinations in Q, K.
        # This is cartesian product matrix of shape (T, T) for every head and batch.
        # The number of features in relationships matrix is B*H*T*T which will be
        # (32 * 8 * 512 * 512) which is 64M. Each feature has 512 / H = 64 dimensions
        # of float32, hence the size is 16G bytes of memory requirement.
        # --------------------------------------------------------------------------------
        relationships: Tensor = calculate_similarities(
            query=q,
            key=k,
            mask=None
        )

        # --------------------------------------------------------------------------------
        # Generate attention values for each element in sequence of length T
        # --------------------------------------------------------------------------------
        attentions: Tensor = calculate_attentions(
            relationships=relationships,
            value=v
        )   # shape: (B,H,T,d)

        # --------------------------------------------------------------------------------
        # Concatenate heads. First (B,H,T,d)->(B,T,H,d) then concatenate to (B,T,M)
        # --------------------------------------------------------------------------------
        attentions = attentions.transpose(2, 1).view(B, T, self.M)

        # --------------------------------------------------------------------------------
        # Last Wo projection
        # --------------------------------------------------------------------------------
        attentions = self.Wo(attentions)
        attentions = self._dropout(attentions)

        return attentions.contiguous()
