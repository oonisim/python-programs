"""Module for the Transformers Model Components.
B: Batch size
T: Sequence length or max token size e.g. 512 for BERT. 'T' because of 'Time steps = Sequence length'
D: Dimensions of the token embedding vector, which is d_model in the paper.
   A token is represented by D number of features or attributes.
   D can be signified as C (Channels).
H: Number of heads in Multi-head attention\
V: Vocabulary size
"""
import math
from typing import (
    Optional
)

import torch
from torch import (
    Tensor,
    nn
)

from transformer.v1.constant import (
    TYPE_FLOAT,
    NUM_CLASSES,
    DIM_MODEL,
    NUM_HEADS,
    MAX_TIME_STEPS,
    POSITION_ENCODE_DENOMINATOR_BASE,
)
from transformer.v1.utility import (
    softmax
)


torch.manual_seed(42)


def initialize_weights(
        module: nn.Module,
        output_projection: bool = False,    # pylint: disable=unused-argument
        i_layer: int = 0,
        d_model: int = DIM_MODEL
):
    """Initialize the module weights
    TODO: Research HuggingFace T5, BERT why they use 0.02 as STD for weight initialization.
    https://stats.stackexchange.com/q/637798/105137
    https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig
    > initializer_range (float, optional, defaults to 0.02)
    > The standard deviation of the truncated_normal_initializer for initializing all weight.

    Depth Initialization - https://aclanthology.org/D19-1083.pdf
    Fixup - https://arxiv.org/pdf/1901.09321.pdf
    T-Fixup https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
    Meta AI Effective Theory of Transformers at Initialization https://arxiv.org/pdf/2304.02034.pdf

    Xavier will use 1/sqrt(D=512) = 0.044 but std=0.02 is used by BERT, BART, GPT.

    DS-init paper (https://aclanthology.org/D19-1083.pdf):
    Improving Deep Transformer with Depth-Scaled Initialization and Merged Attention
    > self-attention sublayer in the encoder is not strong enough to counteract
    > the gradient loss in the feedforward sublayer. That is why BERT and GPT
    > adopt a much smaller standard deviation (0.02) for initialization,
    > in a similar spirit to our solution

    Point-wise feed forward (PwFF) internal dimension is 2048 and 1/sqrt(2048) is 0.022.
    This may be the reason to use 0.02 to make the weight variance consistent throughout
    the layers, but need to be verified. However then, setting the variance of the PwFF
    weight to 0.02 should suffice, not all over the layers.

    TODO: Implement the way to introspect the weight variance during the training.
    (as Andrej Karpathy did but can be instead done by TensorBoard)

    Args:
        module: module whose weights to initialize
        output_projection: TBD
        i_layer: layer index
        d_model: model weight dimension
    """
    # 64 not to be too deep
    assert 0 <= i_layer <= 64, \
        f"expected layer index between {0} and {64}, got [{i_layer}]."
    layer_level: int = i_layer + 1      # to avoid div by 0 at sqrt

    if module.bias is not None:
        torch.nn.init.zeros_(module.bias)

    if isinstance(module, nn.Linear):
        # if output_projection:
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * num_layers))
        # else:
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(
            module.weight, mean=0.0, std=math.sqrt(d_model * layer_level)
        )

    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


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
    _B, _T, _D = x.shape        # pylint: disable=invalid-name
    d_k = _D // h               # pylint: disable=invalid-name
    return x.view(_B, _T, h, d_k).transpose(1, 2)   # (B,h,T,d)


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
    # (B, h, T, d_k) @ (B, h, d_k, T) ---> (B, h, T, T)
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
        similarities: similarity (q to k) matrix to mask of shape (B,H,T,T)
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
        similarities.shape[-2] == similarities.shape[-1] and    # (..., T,T)
        similarities.shape[-1] == mask_matrix.shape[-1]
    )
    # softmax will make the similarity value to 0 when -inf is filled.
    # Mask the similarities (from q to k) matrix:(T,T) to prevent the communications
    # with future time steps by replacing the similarity values with the -inf,
    # meaning there is no relationship from q to k.
    # Then, softmax will make the contribution to zero due to exp(-inf) = 0.
    # This is the same with blocking relations between q and k.
    masked = similarities.masked_fill(mask=mask_matrix, value=float('-inf'))
    return masked


def calculate_attention_values(
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
    > 3.2.3
    > self-attention layers in the decoder allow each position in the decoder to
    > attend to all positions in the decoder up to and including that position.
    > We need to prevent leftward information flow in the decoder to preserve the
    > auto-regressive property. We implement this inside of scaled dot-product
    > attention by masking out (setting to −∞) all values in the input of the
    > softmax which correspond to illegal connections.
    > [NOTE] see mask() function.
    """
    def __init__(self, do_mask: bool, max_time_steps: Optional[int]):
        """
        Args:
            max_time_steps: max sequence length or time steps T
        """
        mask_matrix: Optional[Tensor] = None
        super().__init__()
        if do_mask:
            # shape:(T, T)
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
            return_similarities: bool = False
    ):
        """Calculate the scaled dot product attention.
        Args:
            q: query of shape (B,h,T,d_k)
            k: key of shape (B,h,T,d_k)
            v: value of shape (B,h,T,d_v)
            return_similarities: flag to return similarity (q to k) scores too.
        """
        # --------------------------------------------------------------------------------
        # First MatMul in the Scaled Dot Product Attention to calculate the similarities
        # (attention) matrix between (q,k) for every (q,k) combinations in Q, K.
        # This is cartesian product matrix of shape (T, T) for every head and batch.
        # The number of features in similarities matrix is B*H*T*T which will be
        # (32 * 8 * 512 * 512) which is 64M. Each feature has 512 / H = 64 dimensions
        # of float32, hence the size is 16G bytes of memory requirement.
        # --------------------------------------------------------------------------------
        similarities: Tensor = calculate_dot_product_similarities(
            query=q,
            key=k,
        )   # (B, h, T, T)
        assert torch.all(torch.isfinite(similarities))

        # --------------------------------------------------------------------------------
        # Scale (standardize) the dot product similarity matrix with its standard deviation.
        # --------------------------------------------------------------------------------
        d_k = k.shape[-1]  # head size
        similarities = scale(similarities=similarities, d_k=d_k)

        # --------------------------------------------------------------------------------
        # Mask if required
        # --------------------------------------------------------------------------------
        if self.mask_matrix is not None:
            similarities = mask(
                similarities=similarities,      # shape:(B,h,T,T)
                mask_matrix=self.mask_matrix    # shape:(T,T)
            )

        # --------------------------------------------------------------------------------
        # Softmax normalization so that dim=-1 (q to k similarity) becomes probability.
        # --------------------------------------------------------------------------------
        similarities = softmax(similarities, dim=-1)

        # --------------------------------------------------------------------------------
        # Second MatMul to generate attention value for each token in sequence of length T
        # --------------------------------------------------------------------------------
        attentions: Tensor = calculate_attention_values(
            similarities=similarities,
            values=v
        )   # shape: (B,H,T,d_k)

        return attentions, similarities if return_similarities else attentions


class MultiHeadAttention(nn.Module):
    """
    Class to implement Multi Head Attention (Figure 2 right in the paper).
    Citation:
    > The encoder is composed of a stack of N = 6 identical layers. Each layer has two
    > sub-layers. The first is a multi-head self-attention mechanism, ... To facilitate
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
    """
    @property
    def L(self) -> int:     # pylint: disable=invalid-name
        """Layer number/index"""
        return self._L

    @property
    def T(self) -> int:     # pylint: disable=invalid-name
        """Max time steps or max sequence length"""
        return self._T

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
    def d_k(self) -> int:     # pylint: disable=invalid-name
        """Dimension (number of features) per head."""
        return self._d_k

    def __init__(
            self,
            i_layer: int,
            num_heads: int = NUM_HEADS,
            d_model: int = DIM_MODEL,
            dtype: Tensor.dtype = TYPE_FLOAT,
            do_mask: bool = False,
            max_time_steps: int = MAX_TIME_STEPS,
            bias: bool = True,
    ):
        """Multi Head Attention initialization.
        Args:
            i_layer: layer index (i-th layer from the bottom)
            num_heads: number of attention heads
            d_model: dimensions of the model embedding vector.
            dtype: data type
            do_mask: True when execute masking to not calculate attention with future time steps
            max_time_steps: max sequence length or time steps T.
            bias: Ture if learning the additive bias at the linear layer.
        """
        assert d_model % num_heads == 0, \
            f"d_model:[{d_model}] needs to be divisible by num_heads:[{num_heads}]."

        super().__init__()
        self._L: int = i_layer                  # pylint: disable=invalid-name
        self._T: int = max_time_steps           # pylint: disable=invalid-name
        self._D: int = d_model                  # pylint: disable=invalid-name
        self._H: int = num_heads                # pylint: disable=invalid-name
        self._d_k: int = d_model // num_heads   # pylint: disable=invalid-name

        # To transfer embedded token of dim_input features to Q space of d_model features
        self.Wq: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=d_model,
            out_features=d_model,
            dtype=dtype,
            bias=bias
        )
        initialize_weights(module=self.Wq, i_layer=i_layer, d_model=d_model)
        # To transfer embedded token of dim_input features to K space of d_model features
        self.Wk: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=d_model,
            out_features=d_model,
            dtype=dtype,
            bias=bias
        )
        initialize_weights(module=self.Wk, i_layer=i_layer, d_model=d_model)
        # To transfer embedded token of dim_input features to V space of d_model features
        self.Wv: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=d_model,
            out_features=d_model,
            dtype=dtype,
            bias=bias
        )
        initialize_weights(module=self.Wv, i_layer=i_layer, d_model=d_model)

        # Self attention
        self.scaled_dot_product_attention: nn.Module = ScaledDotProductAttention(
            do_mask=do_mask,
            max_time_steps=max_time_steps
        )

        # Projection to apply to the concatenated output of Self Dot Product Attention
        self.Wo: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=d_model,
            out_features=d_model,
            dtype=dtype,
            bias=bias
        )
        initialize_weights(
            module=self.Wo, i_layer=i_layer, d_model=d_model, output_projection=True
        )

    def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor
    ):
        """Run multi head attention
        Args:
            q: input embedding vectors of shape (B,T,D)
            k: input embedding vectors of shape (B,T,D)
            v: input embedding vectors of shape (B,T,D)

        Returns: Attention values of shape (B,T,D)
        """
        # pylint: disable=invalid-name
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3, \
            "expected q.ndim == 3 and k.ndim == 3 and v.ndim == 3, " \
            f"got {q.ndim}, {k.ndim}, {v.ndim}"
        assert q.shape == k.shape == v.shape, \
            "expected q.shape == k.shape == v.shape, " \
            f"got {q.shape}, {k.shape}, {v.shape}."

        _B, _T, _D = q.shape      # Batch, Tokens (or sequence length), Dimension
        assert _D == self.D, \
            f"input vector dimension is invalid, expected [{self.D}], got [{_D}]."

        # --------------------------------------------------------------------------------
        # Transfer x into Q, K, V spaces. Corresponds to the first 'Linear' layers in
        # the figure 2 of the original paper. In the paper, there are H number of Linear
        # layers Q, K, V respectively, but no need to physically split into H number of
        # Linear layers. Instead, use one Linear layer Wq, Wk, Wv for Q, K, V respectively.
        # --------------------------------------------------------------------------------
        q: Tensor = self.Wq(q)   # Transfer to Q space. Shape=(B, T, D)
        k: Tensor = self.Wk(k)   # Transfer to K space. Shape=(B, T, D)
        v: Tensor = self.Wv(v)   # Transfer to V space. Shape=(B, T, D)
        assert q.shape == (_B, self.T, self.D)
        assert k.shape == (_B, self.T, self.D)

        # --------------------------------------------------------------------------------
        # Split into H segments for multiple heads to attend.
        # --------------------------------------------------------------------------------
        q = split(x=q, h=self.H)    # (B, H, T, d_k)
        k = split(x=k, h=self.H)    # (B, H, T, d_k)
        v = split(x=v, h=self.H)    # (B, H, T, d_k)

        # --------------------------------------------------------------------------------
        # Calculate self attention values
        # --------------------------------------------------------------------------------
        attentions: Tensor = self.scaled_dot_product_attention(q=q, k=k, v=v)
        assert attentions.shape == (_B, self.H, _T, self.d_k)

        # --------------------------------------------------------------------------------
        # Concatenate outputs from heads into the model output with reshape.
        # First (B,H,T,d)->(B,T,H,d) then to (B,T,D).
        # To apply 'view' operation, the Tensor elements need to be stored contiguously in
        # memory. Otherwise:
        # "RuntimeError: view size is not compatible with input tensor's size and strid"
        # --------------------------------------------------------------------------------
        attentions = attentions.transpose(2, 1).contiguous().view(_B, _T, -1)
        assert attentions.shape == (_B, _T, self.D)

        # --------------------------------------------------------------------------------
        # Last Wo Linear projection
        # --------------------------------------------------------------------------------
        attentions = self.Wo(attentions)    # (B,T,D)@(D,D) -> (B,T,D)
        assert attentions.shape == (_B, _T, self.D), \
            f"expected attention shape {(_B, _T, self.D)}, got {attentions.shape}."
        assert torch.all(torch.isfinite(attentions))

        return attentions


class PositionwiseFeedForward(nn.Module):
    """Class to implementation of Position-wise Feed-Forward Networks.
    This is a single hidden layer nural network with ReLU activation.
    """
    def __init__(
            self,
            i_layer: int,
            d_model: int = DIM_MODEL,
            d_ff: int = 2048,
            dtype: Tensor.dtype = TYPE_FLOAT,
            bias: bool = True,
    ):
        """Initialize the class
        Args:
            i_layer: layer index (i-th layer from the bottom)
            d_model: dimensions of the model embedding vector.
            d_ff: dimensions of the hidden layer output vector
            dtype: data type
            bias: True to learn additive bias in the layer.
        """
        super().__init__()
        self.W1: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=d_model, out_features=d_ff, bias=bias, dtype=dtype
        )
        # Weight initialization for ReLU
        torch.nn.init.zeros_(self.W1.bias)
        torch.nn.init.kaiming_normal_(
            self.W1.weight, a=0, mode='fan_in', nonlinearity='relu'
        )
        # TODO: Consider using GELU after verifying the rationale
        self.relu = nn.ReLU()

        self.W2: nn.Module = nn.Linear(     # pylint: disable=invalid-name
            in_features=d_ff, out_features=d_model, bias=bias, dtype=dtype
        )
        initialize_weights(module=self.W2, i_layer=i_layer, output_projection=True)

    def forward(self, x):
        """Feed-forward neural network forward propagation
        Citation:
        > each of the layers in our encoder and decoder contains a fully connected
        > feed-forward network, which is applied to each position separately and
        > identically. This consists of two linear transformations with a ReLU
        > activation in between.
        > Another way of describing this is as two convolutions with kernel size 1.
        > The dimensionality of input and output is dmodel = 512, and the inner-layer
        > has dimensionality of d_ff = 2048.

        Args:
            x: input embedding vector of shape (B,T,D)

        Returns: output embedding vector of shape (B,T,D)
        """
        y = self.W2(self.relu(self.W1(x)))
        assert torch.all(torch.isfinite(y))

        return y


class InputEmbedding(nn.Module):
    """Class to implement the input embedding.
    Citation:
    > 3.4 Embeddings and Softmax:
    > Similarly to other sequence transduction models, we use learned embeddings
    > to convert the input tokens and output tokens to vectors of dimension d_model.
    > ... In the embedding layers, we multiply those weights by sqrt(d_model).

    Note that it is not clear why embedding is multiplied by sqrt(d_model).
    See https://datascience.stackexchange.com/a/87909/68313.
    """
    def __init__(
            self,
            vocabulary_size: int,
            d_model: int = DIM_MODEL,
            dtype: torch.dtype = TYPE_FLOAT
    ):
        """
        Args:
            vocabulary_size: number of vocabularies to encode
            d_model: embedding vector dimension
            dtype: data type of the embedding vector
        """
        super().__init__()
        self._D: int = d_model  # pylint: disable=invalid-name
        self.embedding: nn.Embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=d_model,
            dtype=dtype
        )
        initialize_weights(module=self.embedding)

    def forward(self, indices: Tensor):
        """Encode token indices into token embeddings.
        Args:
            indices: indices to tokens of shape (B, T)
        Returns: Token embeddings of shape (B, T, D)
        """
        # Why multipy by sqrt(D) and increase the variance?
        x = self.embedding(indices) * math.sqrt(self._D)     # x.shape=(B,T,D)
        assert torch.all(torch.isfinite(x))
        return x


class PositionalEncoding(nn.Module):
    """Class to implement the positional encoding.
    Taken from https://nlp.seas.harvard.edu/annotated-transformer/
    The responsibility of this class is to provide position encodings,
    NOT to add them to the token vectors.

    NOTE: DO NOT forget Dropout in Encoder/Decoder classes.
    """
    def __init__(
            self,
            max_time_steps: int = MAX_TIME_STEPS,
            d_model: int = DIM_MODEL,
            dtype: torch.dtype = TYPE_FLOAT
    ):
        """
        Args:
            max_time_steps: max sequence length or time steps T
            d_model: embedding vector dimension
            dtype: data type of the embedding vector
        """
        super().__init__()
        self.D: int = d_model   # pylint: disable=invalid-name

        # Compute the positional encodings once in log space.
        position_encodings = torch.zeros(max_time_steps, d_model, dtype=dtype)  # shape:(T,D)
        positions = torch.arange(0, max_time_steps, dtype=dtype).unsqueeze(1)   # shape:(T,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=dtype) *
            -1 * (math.log(torch.tensor(POSITION_ENCODE_DENOMINATOR_BASE, dtype=dtype)) / d_model)
        )
        position_encodings[:, 0::2] = torch.sin(positions * div_term)
        position_encodings[:, 1::2] = torch.cos(positions * div_term)
        position_encodings = position_encodings.unsqueeze(0)                    # shape:(1,T,D)
        self.register_buffer("position_encodings", position_encodings)
        assert self.position_encodings.shape == (1, max_time_steps, d_model)

    def forward(self, x: Tensor):
        """Positional encoding.
        Args:
            x: token embedding vectors of shape (B,T,D)
        Returns: Position encoding for x as shape (1, T, D)
        """
        assert x.ndim == 3, f"expected x.shape as (B, T, D), got {x.shape}."
        _, _T, _D = x.shape       # pylint: disable=invalid-name
        assert self.D == _D, f"expected the dimension of x element is [{self.D}], got [{_D}]."

        y = self.position_encoding[:, :_T].requires_grad_(False)
        assert torch.all(torch.isfinite(y))
        assert y.shape == (1, _T, _D)
        return y


class Projection(nn.Module):
    """Class to project the predictions to class probabilities.
     """
    def __init__(
            self,
            d_model: int = DIM_MODEL,
            num_classes: int = NUM_CLASSES,
            dtype: Tensor.dtype = TYPE_FLOAT,
            bias: bool = True,
    ):
        super().__init__()
        self.projection: nn.Linear = nn.Linear(
            in_features=d_model,
            out_features=num_classes,
            dtype=dtype,
            bias=bias
        )

        initialize_weights()

    def forward(
            self,
            y: Tensor
    ):
        """Project the prediction embedding to probabilities of vocabularies.
        Args:
            y: prediction of shape (B, T, D)

        Returns: probabilities of shape (B, T, V)
        """
        return
