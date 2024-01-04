"""Module for Transformers constant"""
import numpy as np
import torch

TYPE_FLOAT = torch.float32

NUM_MULTI_ATTENTION_HEADS: int = 8
H: int = NUM_MULTI_ATTENTION_HEADS

DIM_TOKEN: int = 512          # Dimension of the token embedding vector
D: int = DIM_TOKEN            # Abbreviation

DIM_MODEL: int = 512          # Dimension ofd the Transformer encoder vector
DIM_SINGLE_HEAD: int = DIM_MODEL // NUM_MULTI_ATTENTION_HEADS
M: int = DIM_SINGLE_HEAD

DROPOUT_RATIO: float = 0.1


