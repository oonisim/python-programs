"""Module for Transformers constant"""
import torch

ENCODER_MAX_TOKENS: int = 10240
DECODER_MAX_TOKENS: int = 10240

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
MAX_TIME_STEPS: int = 512                       # Max Time Step Sequence (e.g Sentence)
ENCODER_MAX_TIME_STEPS: int = MAX_TIME_STEPS
DECODER_MAX_TIME_STEPS: int = MAX_TIME_STEPS
NUM_LAYERS: int = 6
ENCODER_LAYERS: int = NUM_LAYERS        # Number of layers in the encoder stack.
DECODER_LAYERS: int = NUM_LAYERS        # Number of layers in the decoder stack.

# Dimension of the hidden layer in the position-wise feed forward network (FFN).
# The FFN is structured as:
# x: (B, T, d_model) → W1: (d_model, d_ff) → ReLU → W2: (d_ff, d_model) → out: (B, T, d_model)
# d_ff is purely internal to each FFN block. The input and output are both d_model. So:
#
# - Encoder FFN: d_model → d_ff_encoder → d_model
# - Decoder FFN: d_model → d_ff_decoder → d_model
#
#   These are independent and never interact with each other directly.
#   Cross-attention only sees the d_model-dimensional outputs from the encoder layers,
#   not the FFN hidden states.
#
DIM_PWFF_HIDDEN: int = 2048             # Dimensions of the Feed Forward Hidden layer
ENCODER_PWFF_DIM: int = DIM_PWFF_HIDDEN
DECODER_PWFF_DIM: int = DIM_PWFF_HIDDEN

DROPOUT_RATIO: float = 0.1
ENCODER_DROPOUT_RATIO: float = DROPOUT_RATIO
DECODER_DROPOUT_RATIO: float = DROPOUT_RATIO
EPSILON: float = 1e-6

# Dimension of the model embedding vector.
# This is the dimension of the input and output of all sub-layers in the model.
# In the original paper, the same d_model is used for both encoder and decoder.
# This is because for the cross attention in the decoder:
# - K, V from encoder: projected by Wk(d_model, d_model) and Wv(d_model, d_model)
# - Q from decoder: projected by Wq(d_model, d_model) — expects input dim = d_model
#
# Since both encoder and decoder use the same d_model, the encoder output dimension
# must match the decoder's d_model. If they differed, the K/V projection would fail
# because the encoder memory shape (B, T_e, d_model_encoder) wouldn't match
# Wk.in_features = d_model_decoder.
#
# It can be possible to decouple them with different Q vs K/V projections
# (e.g. Wk(d_model_encoder, d_k) and Wq(d_model_decoder, d_k)). However, this would
# add complexity and is not how the original paper implemented it.
DIM_MODEL: int = 512

# Number of possible next tokens to predict at the final projection layer e.g.
# vocabulary size of the decoder tokenizer and number of classes at the final softmax.
NUM_CLASSES: int = DECODER_MAX_TOKENS

# --------------------------------------------------------------------------------
# Multi Head Attention
# --------------------------------------------------------------------------------
NUM_HEADS: int = 8
