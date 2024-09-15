"""Module to implement Transformer"""
import torch
from torch import (
    Tensor,
    nn
)

from transformer.v1.constant import (
    TYPE_FLOAT,
    NUM_CLASSES,
    DIM_MODEL,
    NUM_LAYERS,
    NUM_HEADS,
    MAX_TIME_STEPS,
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
            vocabulary_size=NUM_CLASSES,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            d_model=DIM_MODEL,
            dtype=TYPE_FLOAT,
            d_ff=2024,
            do_mask=False,
            max_time_steps=MAX_TIME_STEPS,
            bias=True,
            p_drop=0.1
        )

        self.decoder: nn.Module = Decoder(
            vocabulary_size=NUM_CLASSES,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            d_model=DIM_MODEL,
            dtype=TYPE_FLOAT,
            d_ff=2024,
            max_time_steps=MAX_TIME_STEPS,
            bias=True,
            p_drop=0.1
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
        """
        predictions = self.projection(y=self.decoder(y=y, memory=self.encoder(x=x)))
        return torch.argmax(predictions)
