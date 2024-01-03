"""Module for the Transformers Model"""
import torch
import torch.functional as F
import torch.nn as nn


torch.manual_seed(42)


class SelfAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()

