"""Module for the Transformers Model Components.
B: Batch size
T: Sequence length or max token size e.g. 512 for BERT. 'T' because of 'Time steps = Sequence length'
D: Dimensions of the token embedding vector, which is d_model in the paper.
   A token is represented by D number of features or attributes.
   D can be signified as C (Channels).
H: Number of heads in Multi-head attention\
V: Vocabulary size
"""
import logging
import math
from typing import (
    Optional
)

import torch
from torch import (
    Tensor,
    nn
)
from torch.nn.functional import (
    softmax,
)

from .constant import (
    TYPE_FLOAT,
    NUM_CLASSES,
    DIM_MODEL,
    DIM_PWFF_HIDDEN,
    NUM_HEADS,
    MAX_TIME_STEPS,
    POSITION_ENCODE_DENOMINATOR_BASE,
)


#torch.manual_seed(42)
logger: logging.Logger = logging.getLogger(__name__)


def initialize_weights_xavier(
        module: nn.Module,
):
    """Xavier weight initialization"""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            # When a Torch Module is created with bias=False, there is no bias parameter.
            # Nothing needs to be set to 0.
            torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_normal_(module.weight)

    elif isinstance(module, nn.Embedding):
        torch.nn.init.xavier_normal_(module.weight)


def initialize_weights(
        module: nn.Module,
        d_model: int,                       # Must be explicitly specified.
        i_layer: int = 0,
        output_projection: bool = False,    # pylint: disable=unused-argument
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

    Position-wise feed forward (PwFF) internal dimension is 2048 and 1/sqrt(2048) is 0.022.
    This may be the reason to use 0.02 to make the weight variance consistent throughout
    the layers, but need to be verified. However then, setting the variance of the PwFF
    weight to 0.02 should suffice, not all over the layers.

    TODO: Implement the way to introspect the weight variance during the training.
    (as Andrej Karpathy did but can be instead done by TensorBoard)

    Args:
        module: module whose weights to initialize
        d_model: model weight dimension (explicitly provided to avoid unexpected implicit default).
        output_projection: TBD
        i_layer: layer index
    """
    # 64 not to be too deep
    assert 0 <= i_layer <= 64, \
        f"expected layer index between {0} and {64}, got [{i_layer}]."
    layer_level: int = i_layer + 1      # to avoid div by 0 at sqrt

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        # if output_projection:
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * num_layers))
        # else:
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(
            module.weight,
            mean=0.0,
            std=1/math.sqrt(2 * d_model * layer_level)  # x 2 is empirical to get close to 0.02 used in GPT/BERT.
        )
    else:
        msg: str = f"Expected Linear module for weight initialization, got:{type(module)}"
        logger.error(msg)
        raise RuntimeError(msg)


def initialize_embedding_weights(
        module: nn.Module,
):
    """Initialize embedding module weights
    The variance of the embedding vector of d_model dimension has the variance of d_model.
    To keep the variance of the embedding vector output to be 1.0, the standard deviation
    of the weight needs to be 1/sqrt(d_model) = 0.044.
    However, BERT and most modern models use a fixed small standard deviation of 0.02.

    Why 0.02? (The Empirical "Sweet Spot"):
    In the original Transformer, the variance was tied to d_model to keep the signal stable.
    However, as models got deeper (like BERT-Large or GPT), researchers found that
    - Vanishing/Exploding Gradients:
    Deep Transformers are incredibly sensitive during the first few steps of training.
    A smaller standard deviation (0.02) acts as a "cold start" mechanism, keeping the
    initial activations very small.
    - LayerNorm Stability:
    Since BERT applies LayerNorm immediately after the embedding and residual sum,
    the exact variance of the input matters less than the relative scale.
    A smaller initial scale allows the LayerNorm to "take control" of the distribution
    more effectively.
    - Weight Tying Harmony:
    When sharing weights with the output projection, 0.02 keeps the initial logits
    (before training) very flat. This prevents the model from starting with a strong,
    random bias toward specific tokens.

    The Disconnect from d_model:
    If we used 1/sqrt(d) for a tiny model (d=128), the std would be 0.088.
    For a huge model (d=4096), it would be 0.015. BERT's creators found that 0.02 works
    robustly across different sizes (Base and Large), making it a "hyperparameter" rather
    than a "calculated parameter."

    SoTA Choice: Fixed Small Standard Deviation (0.02)
    Most modern models use a fixed range, regardless of d_model. This is because modern
    architectures rely on LayerNorm (specifically Pre-LayerNorm) to handle the scaling.

    Impact on Weight Tying:
    Even though the variance is 0.02^2 instead of 1/d_model, the Weight Tying logic
    still holds. By using bias=False in your projection head and tying it to an embedding
    initialized at std=0.02, the model starts "humble" (low variance, flat probabilities).
    The Embedding and Projection remain symmetry of each other in the vector space.
    """
    if isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    else:
        msg: str = f"Expected Embedding module for embedding initialization, got:{type(module)}"
        logger.error(msg)
        raise RuntimeError(msg)


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
    Note that the sequence length from Encoder  can be different from that of Decoder
    in Cross Attention in the Decoder. Tq may not be equal with Tk.

    Q@K^T = (B, h, Tq, d_k) @ (B, h, d_k, Tk) -> (B, h, Tq, Tk)

    Args:
        query: embedding vector of query of shape (B, h, Tq, d_k)
        key: embedding vector of key of shape (B, h, Tk, d_k)

    Returns: Similarities (closeness) between q and k of shape (B, h, Tq, Tk) where
        last (Tq, Tk) represents relations between all query elements in Tq sequence
        against all key elements in Tk sequence. If T is people in an organization,
        (T,T) represents all (cartesian product) social connections among them.
        The relation considers d_k number of features.
    """
    # --------------------------------------------------------------------------------
    # Relationship between k and q as the first MatMul using dot product similarity:
    # (B, h, Tq, d_k) @ (B, h, d_k, Tk) ---> (B, h, Tq, Tk)
    # --------------------------------------------------------------------------------
    similarities = query @ key.transpose(-2, -1)            # dot product
    return similarities                                     # shape:(B, h, Tq, Tk)


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
    # similarities.new_tensor assures std and similarities will be on the same device.
    # The similarities / std will not cause calculation between mismatched devices.
    std = torch.sqrt(similarities.new_tensor(d_k, dtype=TYPE_FLOAT))   # standard deviation

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
    """Mask the future tokens to prevent future leak.
    Args:
        similarities: similarity (q to k) matrix to mask of shape (B,H,Tq,Tk)
        mask_matrix: boolean matrix of which elements in (Tk,Tk) to mask fill.

    Returns: masked similarity matrix
    """
    # --------------------------------------------------------------------------------
    # mask to make uni-direction (left to right only) for algorithm such as GPT.
    # Skip masking for bi-directional e.g .BERT,
    # --------------------------------------------------------------------------------
    # exp(-inf) = 0 masks the similarities so that it will be uni-directional.
    if similarities.ndim != 4:   # (B,H,Tq,Tk)
        raise RuntimeError(f"Expected similarities of shape (B,H,Tq,Tk), got {similarities.shape}")
    if similarities.device != mask_matrix.device:
        raise RuntimeError(f"similarities on {similarities.device} but mask_matrix on {mask_matrix.device}")


    # --------------------------------------------------------------------------------
    # Adjust the Future Mask shape to match the actual sequence length _Tk.
    # If the mask is larger than the actual sequence length _Tk, cut to _Tk x _Tk.
    # Otherwise, shape mismatch in masked_fill and (_T, _T) mask can’t be broadcast
    # to (B, H, _Tk, _Tk).
    #
    # In causal self-attention of the Decoder, each query in Q only see the past keys
    # in K and must not see future keys. Because LM predicts next from the past only.
    # --------------------------------------------------------------------------------
    _Tk = similarities.shape[-1]    # pylint: disable=invalid-name
    if mask_matrix.shape[-1] != _Tk:
        mask_matrix = mask_matrix[:_Tk, :_Tk]

    # --------------------------------------------------------------------------------
    # Mask the similarities (from q to k) matrix:(Tk,Tk) to prevent q looking into
    # the future time steps in the sequence. This is to prevent the future leak in
    # the uni-directional attention such as GPT.
    # By replacing similarity values of the future time steps with the -inf,
    # there is no relationship from q to k in the similarity matrix.
    # Then, softmax will make the contribution to zero due to exp(-inf) = 0.
    # This is the same with blocking relations between q and k.
    # --------------------------------------------------------------------------------
    masked = similarities.masked_fill(mask=mask_matrix, value=float('-inf'))
    return masked


def calculate_attention_values(
        similarities: Tensor,
        values: Tensor
):
    """
    For every q element, create a Bag of Words that encodes the relationships with
    other elements (including itself) in T, using (q,k) relationship value as the
    strength of the relationships.

    Note that in the Cross Attention in the Decoder, sequence length of Q (Tq) may
    be different from that from Encoder (Tk = T_v).

    The similarities matrix has the shape (B, h, Tq, Tk). V has (B, h, Tk, d_v).
    (B, h, Tq, Tk) @ (B, h, Tk, d_v) -> (B, h, Tq, d_v)

    Citation:
    > On each of these projected versions of queries, keys and values we then perform
    > the attention function in parallel, yielding d_v-dimensional output values.

    ```
    bows = []
    for row in similarities:                    # similarity matrix of shape (Tq,Tk)
        bow = sum([                             # bow:shape(d_v,)
            # each column in row is (q,k) similarity score s
            s*v for (s,v) in zip(row,values)    # k:shape(), v:shape(d_v,)
=        ])
        bows.append(bow)                        # bows:shape(Tq,d_v)
    ```

    Args:
        similarities: q to k relationship strength matrix of shape (B, h, Tq, Tk)
        values: elements of sequence with length Tk of shape (B, h, Tk, d_v) where Tk = T_v.

    Returns: Bag of Words for every q element of shape (B, h, Tq, d_v)
    """
    if similarities.device != values.device:
        raise RuntimeError(f"similarities on {similarities.device} but values on {values.device}")

    return similarities @ values     # (B,h,Tq,Tk) @ (B,h,Tk,d_v) -> (B,h,Tq,d_v)


class LayerNormalization(nn.Module):    # pylint: disable=too-few-public-methods
    """Class to implement LayerNormalization"""
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        """Initialization of the class"""
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))     # alpha is a learnable parameter
        self.beta = nn.Parameter(torch.zeros(features))     # bias is a learnable parameter

    # pylint: disable=anomalous-backslash-in-string
    def forward(self, x: Tensor):
        r"""Run Layer Normalization
        https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        $$ y = \gamma \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} + \beta $$

        Torch.var() or std() uses Bessel's correction by default, which divides by (N-1).
        Layer Normalization must use the population N because we need Var(Xnormalized)=1
        that is the objective of Layer Normalization.

        Args:
            x: input

        Returns: y as normalized x
        """
        # x: (batch, seq_len, hidden_size) = (B, T, D)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)                     # (B, T, 1)
        # Keep the dimension for broadcasting
        variance = x.var(dim=-1, keepdim=True, correction=0)    # (B, T, 1)
        # eps is added inside sqrt for numerical stability when variance is near zero
        y = self.gamma * (x - mean) / torch.sqrt(variance + self.eps) + self.beta
        return y


class ScaledDotProductAttention(nn.Module):     # pylint: disable=too-few-public-methods
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
            # This is for Decoder self attention to prevent future leak.
            # The mask matrix is the same for all batches and heads.
            # shape:(T, T)
            self.do_mask: bool = do_mask
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
        Note that in the Cross Attention in Encoder, the sequence length may be different
        from that from the Encoder. Hence, be specific with (Tk = T_v), and Tq

        Args:
            q: query of shape (B,h,Tq,d_k)
            k: key of shape (B,h,Tk,d_k)
            v: value of shape (B,h,Tk,d_v)
            return_similarities: flag to return similarity (q to k) scores too.
        """
        if q.ndim != 4:
            raise RuntimeError(f"Expected query of shape (B,h,Tq,d_k), got {q.shape}")
        if not q.device == k.device == v.device:
            raise RuntimeError(f"Inconsistent devices: Q on {q.device}, K on {k.device}, V on {v.device}.")

        _B, _H, _Tq, _ = q.shape    # pylint: disable=invalid-name
        _Tk = k.shape[-2]           # pylint: disable=invalid-name

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
        assert similarities.shape == (_B, _H, _Tq, _Tk)
        # TODO:This is an expensive operation. Consider conditional switch.
        assert torch.all(torch.isfinite(similarities))

        # --------------------------------------------------------------------------------
        # Scale (standardize) the dot product similarity matrix with its standard deviation.
        # --------------------------------------------------------------------------------
        d_k = k.shape[-1]  # head size
        similarities = scale(similarities=similarities, d_k=d_k)
        assert similarities.shape == (_B, _H, _Tq, _Tk)
        assert torch.all(torch.isfinite(similarities))

        # --------------------------------------------------------------------------------
        # When Mask if required for causal attention, mask the similarities matrix to
        # prevent the attention to future time steps. Masking fills the similarity value
        # with -inf so that the softmax will make the contribution to zero as exp(-inf) = 0.
        # This is the same as blocking relations between q and k.
        # --------------------------------------------------------------------------------
        if self.mask_matrix is not None:
            # In the causal attention, the sequence length of q and k should be the same
            # because the attention is only within the same sequence.
            # In the cross attention, the sequence length of Tq and Tk can be different
            # because the attention is between different sequences.
            if _Tk != _Tq:
                if self.do_mask:
                    # Decoder may have different sequence length between Q and K because of
                    # Encoder may generate K,V whose sequence length Tk is different from
                    # the Tq of the Decoder at Cross Attention. However, currently does not
                    # support such case. Need future customisation to accept it.
                    raise ValueError(
                        "Decoder (do_mask=True) does not support different sequence lengths "
                        f"between Q and K now, got Tq:[{_Tq}] != Tk:[{_Tk}]."
                    )

                # do_mask is False means Encoder self attention, and the sequence length T
                # of Q and K should be the same.
                raise RuntimeError(
                    "Encoder Self-Attention (do_mask=False) requires the same sequence "
                    f"length between Q and K, got Tq:[{_Tq}] and Tk:[{_Tk}]."
                )

            similarities = mask(
                similarities=similarities,      # shape:(B,h,T,T)
                mask_matrix=self.mask_matrix    # shape:(T,T)
            )
            assert similarities.shape == (_B, _H, _Tq, _Tk)
            # mask() set -inf
            # assert torch.all(torch.isfinite(similarities))

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
        assert attentions.shape == (_B, _H, _Tq, d_k)

        if return_similarities:
            return attentions, similarities

        return attentions


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

    def __init__(   # pylint: disable=too-many-arguments
            self,
            i_layer: int = 0,
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
        Note: the sequence lengths of Encoder and Decoder can be different Cross Attention.
        Q: (B, Tq, D) from decoder
        K, V: (B, Tk, D) from encoder

        After projections:
        Q: (B, H, Tq, d_k)
        K: (B, H, Tk, d_k)
        V: (B, H, Tk, d_v)

        In this implementation, d_k == d_v == D / H.

        Then:
        (B, H, Tq, d_k) @ (B, H, d_k, Tk) -> (B, H, Tq, Tk)
        (B, H, Tq, Tk) @ (B, H, Tk, d_v) -> (B, H, Tq, d_v)

        Args:
            q: input embedding vectors of shape (B,Tq,D)
            k: input embedding vectors of shape (B,Tk,D)
            v: input embedding vectors of shape (B,Tk,D)


        Returns: Attention values of shape (B,T,D)
        """
        # pylint: disable=invalid-name
        if not (q.ndim == 3 and k.ndim == 3 and v.ndim == 3):
            raise ValueError(
                f"expected q.ndim == 3 and k.ndim == 3 and v.ndim == 3, got {q.ndim}, {k.ndim}, {v.ndim}."
            )
        if not q.shape[0] == k.shape[0] == v.shape[0]:
            raise ValueError(
                f"expected same batch size for q:[{q.shape[0]}], k:[{k.shape[0]}], v:[{v.shape[0]}]."
            )
        if not q.shape[2] == k.shape[2] == v.shape[2]:
            raise ValueError(
                f"expected same feature dimension for q:[{q.shape[2]}], k:[{k.shape[2]}], v:[{v.shape[2]}]."
            )
        if not q.device == k.device == v.device:
            raise RuntimeError(f"Inconsistent devices: Q on {q.device}, K on {k.device}, V on {v.device}.")

        _B, _Tq, _D = q.shape      # Batch, Tokens (or sequence length), Dimension
        _Tk = k.shape[1]
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
        assert q.shape == (_B, _Tq, self.D)
        assert k.shape == (_B, _Tk, self.D)
        assert v.shape == (_B, _Tk, self.D)

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
        assert attentions.shape == (_B, self.H, _Tq, self.d_k)

        # --------------------------------------------------------------------------------
        # Concatenate outputs from heads into the model output with reshape.
        # First (B,H,T,d)->(B,T,H,d) then to (B,T,D).
        # To apply 'view' operation, the Tensor elements need to be stored contiguously in
        # memory. Otherwise:
        # "RuntimeError: view size is not compatible with input tensor's size and strid"
        # --------------------------------------------------------------------------------
        attentions = attentions.transpose(2, 1).contiguous().view(_B, _Tq, -1)
        assert attentions.shape == (_B, _Tq, self.D)

        # --------------------------------------------------------------------------------
        # Last Wo Linear projection
        # --------------------------------------------------------------------------------
        attentions = self.Wo(attentions)    # (B,T,D)@(D,D) -> (B,T,D)
        assert attentions.shape == (_B, _Tq, self.D), \
            f"expected attention shape {(_B, _Tq, self.D)}, got {attentions.shape}."
        assert torch.all(torch.isfinite(attentions))

        return attentions


class PositionwiseFeedForward(nn.Module):   # pylint: disable=too-few-public-methods
    """Class to implementation of Position-wise Feed-Forward Networks.
    This is a single hidden layer nural network with ReLU activation.
    """
    def __init__(   # pylint: disable=too-many-arguments
            self,
            i_layer: int = 0,
            d_model: int = DIM_MODEL,
            d_ff: int = DIM_PWFF_HIDDEN,
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
        initialize_weights(module=self.W2, i_layer=i_layer, d_model=d_model, output_projection=True)

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
        if not x.device == self.W1.weight.device == self.W2.weight.device:
            raise RuntimeError(
                f"x is on {x.device} but W1 on {self.W1.weight.device} W2 on {self.W2.weight.device}."
            )

        y = self.W2(self.relu(self.W1(x)))
        assert torch.all(torch.isfinite(y))

        return y


class InputEmbedding(nn.Module):    # pylint: disable=too-few-public-methods
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
        initialize_embedding_weights(module=self.embedding)

    def forward(self, indices: Tensor):
        """Encode token indices into token embeddings.
        Args:
            indices: indices to tokens of shape (B, T)
        Returns: Token embeddings of shape (B, T, D)
        """
        # Why multipy by sqrt(D) and increase the variance?
        # Section 3.4 of "Attention is All You Need":
        # "In the embedding layers, we multiply those weights by √d_model."
        # In the original paper, √d_model scaling stabilized the first attention
        # layer as it was Post-LayerNorm Transformers. Pre-LayerNorm normalization
        # removed the need for this scaling.
        x = self.embedding(indices) * math.sqrt(self._D)     # x.shape=(B,T,D)
        assert torch.all(torch.isfinite(x))
        return x


class PositionalEncoding(nn.Module):    # pylint: disable=too-few-public-methods
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

        if self.position_encodings.device != x.device:
            raise RuntimeError(
                f"position_encodings is on {self.position_encodings.device} but x on {x.device}."
            )

        y = self.position_encodings[:, :_T].requires_grad_(False)
        assert torch.all(torch.isfinite(y))
        assert y.shape == (1, _T, _D)
        return y


class Projection(nn.Module):    # pylint: disable=too-few-public-methods
    """
    Class to project the predictions of shape (B, T, D) to vocabulary probabilities of shape (B, T).

    Returns log-probabilities by default. Then Do NOT pass the Projection output to CrossEntropyLoss,
    as CrossEntropyLoss expects raw logits and internally applies log-softmax + NLLLoss.
    Use a loss function that expects log-probabilities, e.g. torch.nn.NLLLoss (negative log-likelihood).
    Or, use return_logits=True to return raw logits and pass the output to CrossEntropyLoss.
    """
    def __init__(
            self,
            d_model: int = DIM_MODEL,
            num_classes: int = NUM_CLASSES,
            dtype: Tensor.dtype = TYPE_FLOAT,
            bias: bool = True,
    ):
        """
        Args:
            d_model: dimension of the model embedding vector
            num_classes: number of classes to predict
            dtype: data type of the projection layer weight
            bias: True to learn additive bias in the projection layer.
        """
        super().__init__()
        self.projection: nn.Linear = nn.Linear(
            in_features=d_model,
            out_features=num_classes,
            dtype=dtype,
            bias=bias
        )
        initialize_weights(self.projection, d_model=d_model)

    def forward(
            self,
            y: Tensor,
            return_logits: bool = False
    ):
        """
        Project the prediction embedding to probabilities or logits of vocabularies.

        Args:
            y: prediction of shape (B, T, D)
            return_logits: If True, return raw logits suitable for CrossEntropyLoss.
                           If False (default), return log-probabilities suitable for NLLLoss.

        Returns: Tensor of shape (B, T, V) — logits or log-probabilities depending on return_logits.
        """
        logits: Tensor = self.projection(y)
        if return_logits:
            return logits
        # Use log_softmax for numerical stability as it prevents overflow issues
        # by avoiding direct computation of exponential.
        return torch.log_softmax(logits, dim=-1, dtype=TYPE_FLOAT)
