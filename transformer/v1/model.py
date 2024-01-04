"""Module for the Transformers Model"""
import torch
import torch.nn as nn

from transformer.v1.constant import (
    TYPE_FLOAT,
    H, NUM_MULTI_ATTENTION_HEADS,
    DIM_MODEL,
    D, DIM_TOKEN,
)
from transformer.v1.utility import (
    get_device,
    clone_module,
    softmax
)


torch.manual_seed(42)


class MultiHeadAttention(nn.Module):
    @property
    def D(self) -> int:
        """Input vector dimension"""
        return self._D

    @property
    def M(self) -> int:
        """Output attention vector dimension"""
        return self._M

    @property
    def H(self) -> int:
        """Number of attention heads"""
        return self._M

    def __init__(
            self,
            num_heads: int = NUM_MULTI_ATTENTION_HEADS,
            dim_input: int = DIM_TOKEN,
            dim_output: int = DIM_MODEL,
            bias: bool = False,
            dropout_ratio: float = 0.1
    ):
        """Class to implement Multi Head Attention
        Args:
            num_heads: number of attention heads
            dim_input: input embedding vector dimension
            dim_output: output attention vector dimension
            bias: if learn additive bias at the linear layer.
        """
        super().__init__()
        self._H: int = num_heads
        self._D: int = dim_input
        self._M: int = dim_output

        # To transfer embedded token of dim_input features to Q space of dim_output features
        self._Wq: nn.Module = nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            dtype=TYPE_FLOAT,
            bias=bias
        )
        # To transfer embedded token of dim_input features to K space of dim_output features
        self._Wk: nn.Module = nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            dtype=TYPE_FLOAT,
            bias=bias
        )
        # To transfer embedded token of dim_input features to V space of dim_output features
        self._Wv: nn.Module = nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            dtype=TYPE_FLOAT,
            bias=bias
        )

        # Project to apply to the concatenated output of Self Dot Product Attention
        self._Wo: nn.Module = nn.Linear(
            in_features=dim_output,
            out_features=dim_output,
            dtype=TYPE_FLOAT,
            bias=bias
        )

        self._dropout: nn.Module = nn.Dropout(
            p=dropout_ratio
        )

    @staticmethod
    def calculate_relationships(
            query,
            key,
            mask=None,
    ):
        """
        Calculate relationship scores between query and keys using dot product similarity.

        Args:
            query: embedding vector of query of shape (B, H, T, d)
            key: embedding vector of key of shape (B, H, T, d)
            mask: mask value to prevent calculating relation with future time step

        Returns: Relationship (closeness) between q and k of shape (B, H, T, T) where
                last (T, T) represents relations between all query elements in T sequence
                against all key elements in T sequence. If T is people in an organization,
                (T,T) represents all (cartesian product) social connections among them.
                The relation considers d number of features.
        """
        d_k = key.size[-1]                                      # head size

        # --------------------------------------------------------------------------------
        # Relationship between k and q as the first MatMul using dot product similarity:
        # (B, H, T, d_k) @ (B, H, d_k, T) ---> (B, H, T, T)
        # --------------------------------------------------------------------------------
        relationships = query @ key.transpose(-2, -1)           # dot product

        # --------------------------------------------------------------------------------
        # Scaling factor to standardize (div by standard deviation) the product q@k.T
        # of two zero centered normal distributions q, k. The variance of the product
        # is head_size d_k. See https://stats.stackexchange.com/a/52699/105137.
        # --------------------------------------------------------------------------------
        std = torch.sqrt(torch.tensor(d_k, dtype=TYPE_FLOAT))   # standard deviation

        # --------------------------------------------------------------------------------
        # Scale relationships of each head by std so that the variance is approx 1.
        # Scaling regularize the softmax output so as not to overfit to features, by which
        # features in query and key can relate among themselves better.
        # Otherwise, features with higher value will be peaked by softmax, (which is good
        # for use as classification head but not for Bag of Words to incorporate features
        # to make them related), hence only specific features in query and key will be
        # connected.
        # --------------------------------------------------------------------------------
        relationships = relationships / std                     # scaled dot product

        # --------------------------------------------------------------------------------
        # mask to make uni-direction (left to right only) for algorithm such as GPT.
        # Skip masking for bi-directional e.g .BERT,
        # --------------------------------------------------------------------------------
        if mask is not None:
            # TODO: Verify if the logic is correct.
            relationships = relationships.masked_fill(mask == 0, float('-inf'))

        # --------------------------------------------------------------------------------
        # Normalize by softmax.
        # exp(-inf) = 0 masks the relationships so that it will be uni-directional.
        # --------------------------------------------------------------------------------
        relationships = softmax(relationships, dim=-1)

        return relationships                                    # shape:(B, H, T, T)

    @staticmethod
    def calculate_attentions(
            relationships,
            value
    ):
        """
        For every q element, create a Bag of Words that encodes the relationships with
        other elements (including itself) in T, using (q,k) relationship value as the
        strength of the relationships.

        ```
        bows = []
        for row in relationships:                   # relationship is matrix of shape (T,T)
            bow = sum([                             # bow:shape(d,)
                k*v for (k,v) in zip(row,value)     # k:shape(), v:shape(d,)
            ])
            bows.append(bow)                        # bows:shape(T,d)
        ```

        Args:
            relationships: q to k relationship strength matrix of shape (B, H, T, T)
            value: elements of sequence with length T of shape (B, H, T, d)

        Returns: Bag of Words for every q element of shape (B, H, T, d)
        """
        return relationships @ value     # (B,H,T,T) @ (B,H,T,d) -> (B,H,T,d)

    def run_multi_head_attentions(
            self,
            x
    ):
        B, T, _D = x.shape      # Batch, Tokens (or sequence length), Dimension
        assert _D == self.D, \
            f"input vector dimension is invalid, expected [{self.D}], got [{_D}]."

        q: torch.Tensor = self.Wq(x)   # Transfer to Q space. Shape=(B, T, M)
        k: torch.Tensor = self.Wk(x)   # Transfer to K space. Shape=(B, T, M)
        v: torch.Tensor = self.Wv(x)   # Transfer to V space. Shape=(B, T, M)
        assert q.shape == (self.B, self.T, self.M)
        assert k.shape == (self.B, self.T, self.M)

        # --------------------------------------------------------------------------------
        # Split Q, K into multiple heads
        # --------------------------------------------------------------------------------
        q = q.view(B, T, self.H, self.M // self.H).transpose(1, 2)  # (B, H, T, d_k)
        k = k.view(B, T, self.H, self.M // self.H).transpose(1, 2)  # (B, H, T, d_k)
        v = v.view(B, T, self.H, self.M // self.H).transpose(1, 2)  # (B, H, T, d_k)

        # --------------------------------------------------------------------------------
        # Build relationships matrix between (q,k) for (q,k) combinations in Q, K.
        # This is cartesian product matrix of shape (T, T) for every head and batch.
        # The number of features in relationships matrix is B*H*T*T which will be
        # (32 * 8 * 512 * 512) which is 64M. Each feature has 512 / H = 64 dimensions
        # of float32, hence the size is 16G bytes of memory requirement.
        # --------------------------------------------------------------------------------
        relationships: torch.Tensor = self.calculate_relationships(
            query=q,
            key=k,
            mask=None
        )

        # --------------------------------------------------------------------------------
        # Generate attention values for each element in sequence of length T
        # --------------------------------------------------------------------------------
        attentions: torch.Tensor = self.calculate_attentions(
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

