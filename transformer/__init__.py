"""Transformer package - Model architecture and training infrastructure.

Directory structure:
- src/model/: Model architecture components (Transformer, LanguageModel, etc.)
- src/training/: Training infrastructure (Trainer, callbacks, data loaders)
- src/test/: Test suite
- run/: Execution scripts (with PYTHONPATH set to src/)

To use this package, ensure PYTHONPATH includes the src/ directory:
  export PYTHONPATH=/path/to/transformer/src
"""

import sys
from pathlib import Path

# Add src directory to path if not already there
_src_dir = Path(__file__).parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

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
