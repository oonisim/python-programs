"""Module to implement Transformer"""
import torch
from torch import (
    Tensor,
    nn
)

from .constant import (
    TYPE_FLOAT,
    NUM_ENCODER_TOKENS,
    NUM_DECODER_TOKENS,
    NUM_CLASSES,
    MAX_TIME_STEPS,
    DIM_MODEL,
    DIM_PWFF_HIDDEN,
    NUM_LAYERS,
    NUM_HEADS,
    DROPOUT_RATIO,
)
from common import (
    Projection,
)
from encoder import (
    Encoder
)
from decoder import (
    Decoder
)


class Transformer(nn.Module):
    """Class to implement the Google Transformer Model"""
    def __init__(self):
        super().__init__()

        self.encoder: nn.Module = Encoder(
            vocabulary_size=NUM_ENCODER_TOKENS,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            d_model=DIM_MODEL,
            dtype=TYPE_FLOAT,
            d_ff=DIM_PWFF_HIDDEN,
            do_mask=False,
            max_time_steps=MAX_TIME_STEPS,
            bias=True,
            p_drop=DROPOUT_RATIO
        )

        self.decoder: nn.Module = Decoder(
            vocabulary_size=NUM_DECODER_TOKENS,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            d_model=DIM_MODEL,
            dtype=TYPE_FLOAT,
            d_ff=DIM_PWFF_HIDDEN,
            max_time_steps=MAX_TIME_STEPS,
            bias=True,
            p_drop=DROPOUT_RATIO
        )

        self.projection: nn.Module = Projection(
            d_model=NUM_HEADS,
            num_classes=NUM_CLASSES,
            dtype=TYPE_FLOAT,
            bias=True
        )

    def forward(
            self,
            x: Tensor,
            y: Tensor
    ) -> Tensor:
        """Run Transformer encoder/decoder
        Args:
            x: Encoder source sequences as token indices of shape (B, T).
            y: Decoder target sequences as token indices of shape (B, T).

        Returns: Indices to vocabularies for the predicted tokens.
        """
        assert x.ndim == 2, f"expected x.shape (B, T), got {x.shape}."
        assert y.ndim == 2, f"expected y.shape (B, T), got {y.shape}."

        predictions: Tensor = self.projection(y=self.decoder(y=y, memory=self.encoder(x=x)))
        predictions = torch.argmax(predictions)
        assert predictions.shape == x.shape

        return predictions
