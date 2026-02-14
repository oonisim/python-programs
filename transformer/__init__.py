"""Transformer package - Model architecture and training infrastructure.

Directory structure:
- model/: Model architecture components (Transformer, LanguageModel, etc.)
- training/: Training infrastructure (Trainer, callbacks, data loaders)
- test/: Test suite
"""

# Re-export from model and training subdirectories
from model.constant import (
    TYPE_FLOAT,
    NUM_LAYERS,
    NUM_HEADS,
    DIM_MODEL,
    DROPOUT_RATIO
)
from model.common import (
    initialize_weights,
    calculate_dot_product_similarities,
    calculate_attention_values,
    split,
    scale,
    mask,
    MultiHeadAttention,
    ScaledDotProductAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
)
from model.encoder import (
    EncodeLayer,
    Encoder,
)
from model.decoder import (
    DecodeLayer,
    Decoder
)
from model.model import Transformer
from model.lm import LanguageModel

# Note: Training classes are not imported here to avoid loading heavy dependencies
# like TensorBoard when only model components are needed.
# Import them explicitly: from training.trainer import Trainer, etc.
