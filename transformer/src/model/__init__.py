"""Model architecture components.

This module contains the transformer model architecture:
- common.py: Core components (MultiHeadAttention, FeedForward, etc.)
- encoder.py: Encoder stack
- decoder.py: Decoder stack
- model.py: Full encoder-decoder Transformer
- lm.py: Language model (decoder-only)
- constant.py: Model constants
"""

# Core components
from .common import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    InputEmbedding,
    Projection,
    LayerNormalization,
    ScaledDotProductAttention,
)

# Model architectures
from .encoder import Encoder, EncodeLayer
from .decoder import Decoder, DecodeLayer
from .model import Transformer
from .lm import LanguageModel

# Constants
from .constant import *

__all__ = [
    # Core components
    'MultiHeadAttention',
    'PositionwiseFeedForward',
    'PositionalEncoding',
    'InputEmbedding',
    'Projection',
    'LayerNormalization',
    'ScaledDotProductAttention',
    # Layers
    'EncodeLayer',
    'DecodeLayer',
    # Architectures
    'Encoder',
    'Decoder',
    'Transformer',
    'LanguageModel',
]
