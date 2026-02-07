"""Module for Transformers constant"""
import torch

NUM_ENCODER_TOKENS: int = 10240
NUM_DECODER_TOKENS: int = 10240

# --------------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------------
TYPE_FLOAT: torch.Tensor.dtype = torch.float32

# --------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------
POSITION_ENCODE_DENOMINATOR_BASE: int = 10000

# --------------------------------------------------------------------------------
# Network Hyper Parameters
# --------------------------------------------------------------------------------
MAX_TIME_STEPS: int = 512       # Max Time Step Sequence (e.g Sentence)
NUM_CLASSES: int = NUM_ENCODER_TOKENS
NUM_LAYERS: int = 6
DIM_MODEL: int = 512            # Dimensions of the model embedding vector
DIM_PWFF_HIDDEN: int = 2048     # Dimensions of the Feed Forward Hidden layer
DROPOUT_RATIO: float = 0.1
EPSILON: float = 1e-6

# --------------------------------------------------------------------------------
# Multi Head Attention
# --------------------------------------------------------------------------------
NUM_HEADS: int = 8
