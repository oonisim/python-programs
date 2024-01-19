"""Module for the Transformers Model
B: Batch size
T: Sequence length or max token size e.g. 512 for BERT. 'T' because of 'Time steps = Sequence length'
H: Number of heads in Multi-head attention
"""
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
        x: Tensor,          # pylint: disable=invalid-name
        h: int              # pylint: disable=invalid-name
) -> Tensor:
    """Split an embedding vector into h segments where each segment has d dimensions
    and gets attended by an attention head. Instead of physically split, reshape the
    embedding vector from (B,T,M) into (B,h,T,d) so that each attention head
    attends (T,d).

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
) -> Tensor:
    """
    Args:
        similarities: matrix to mask

    Returns: masked similarity matrix
    """
    # --------------------------------------------------------------------------------
    # mask to make uni-direction (left to right only) for algorithm such as GPT.
    # Skip masking for bi-directional e.g .BERT,
    # --------------------------------------------------------------------------------
    # TODO: Verify if the logic is correct.
    # exp(-inf) = 0 masks the similarities so that it will be uni-directional.
    masked = similarities.masked_fill(mask == 0, float('-inf'))
    return masked


def calculate_attentions(
        similarities,
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
        similarities: q to k relationship strength matrix of shape (B, h, T, T)
        value: elements of sequence with length T of shape (B, h, T, d_v)

    Returns: Bag of Words for every q element of shape (B, h, T, d_v)
    """
    return similarities @ value     # (B,h,T,T) @ (B,h,T,d_v) -> (B,h,T,d_v)


class ScaledDotProductAttention(nn.Module):
    """
    Class to implement Scaled Dot Product Attention (Figure 2 left in the paper).
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(
            q: Tensor,
            k: Tensor,
            v: Tensor,
            masking: bool
    ):
        """Calculate the scaled dot product attention.
        Args:
            q: query of shape (B,h,T,d)
            k: key of shape (B,h,T,d)
            v: value of shape (B,h,T,d)
            masking: flat to prevent calculating relation with future time step

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
        if masking:
            similarities = mask(similarities=similarities)

        # --------------------------------------------------------------------------------
        # Normalize by softmax.
        # --------------------------------------------------------------------------------
        similarities = softmax(similarities, dim=-1)

        # --------------------------------------------------------------------------------
        # Generate attention values for each element in sequence of length T
        # --------------------------------------------------------------------------------
        attentions: Tensor = calculate_attentions(
            similarities=similarities,
            value=v
        )   # shape: (B,H,T,d)

        return attentions


class MultiHeadAttention(nn.Module):
    """
    Class to implement Multi Head Attention (Figure 2 right in the paper).
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
        """Dimensions (number of features) of the embedding vector (Q,K,V) of a token
        fed into the Multi Head Attention.
        """
        return self._D

    @property
    def M(self) -> int:     # pylint: disable=invalid-name
        """Dimensions of the embedding vector output from the Multi Head Attention."""
        return self._M

    @property
    def H(self) -> int:     # pylint: disable=invalid-name
        """Number of attention heads"""
        return self._H

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
        self._scaled_dot_product_attention: nn.Module = ScaledDotProductAttention()
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

        # --------------------------------------------------------------------------------
        # Transfer x into Q, K, V spaces. Corresponds to the first 'Linear' layers in
        # the figure 2 of the original paper. In the paper, there are H number of Linear
        # layers Q, K, V respectively, but no need to physically split into H number of
        # Linear layers. Instead, use one Linear layer Wq, Wk, Wv for Q, K, V respectively.
        # --------------------------------------------------------------------------------
        q: Tensor = self.Wq(x)   # Transfer to Q space. Shape=(B, T, M)
        k: Tensor = self.Wk(x)   # Transfer to K space. Shape=(B, T, M)
        v: Tensor = self.Wv(x)   # Transfer to V space. Shape=(B, T, M)
        assert q.shape == (self.B, self.T, self.M)
        assert k.shape == (self.B, self.T, self.M)

        # --------------------------------------------------------------------------------
        # Split into H segments for multiple heads to attend.
        # --------------------------------------------------------------------------------
        q = split(x=q, h=self.H)    # (B, H, T, d)
        k = split(x=k, h=self.H)    # (B, H, T, d)
        v = split(x=v, h=self.H)    # (B, H, T, d)

        # --------------------------------------------------------------------------------
        # Calculate self attention values
        # --------------------------------------------------------------------------------
        attentions: Tensor = self._scaled_dot_product_attention(q=q, k=k, v=v)
        assert attentions.shape == (B, self.H, T, self.M/self.H)

        # --------------------------------------------------------------------------------
        # Concatenate outputs from heads. First (B,H,T,d)->(B,T,H,d) then concatenate to (B,T,M)
        # --------------------------------------------------------------------------------
        attentions = attentions.transpose(2, 1).view(B, T, -1)
        assert attentions.shape == (B, T, self.M)

        # --------------------------------------------------------------------------------
        # Last Wo Linear projection
        # --------------------------------------------------------------------------------
        attentions = self.Wo(attentions)    # (B,T,M)@(M,M) -> (B,T,M)
        attentions = self._dropout(attentions)
        assert attentions.shape == (B, T, self.M)

        return attentions.contiguous()
