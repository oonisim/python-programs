"""Module for Transformers constant"""
import numpy as np
import torch

TYPE_FLOAT = torch.float32

NUM_HEADS: int = 8
H: int = NUM_HEADS

DIM_MODEL: int = 512          # Dimension ofd the Transformer encoder vector

DROPOUT_RATIO: float = 0.1

NUM_LAYERS: int = 8
N: int = NUM_LAYERS

