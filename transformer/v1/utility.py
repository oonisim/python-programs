"""Module for Transformers utilities"""
import copy
import torch
import torch.nn as nn
from torch.nn.functional import (
    softmax,
)


def clone_module(
        module: nn.Module,
        num_modules: int
) -> nn.ModuleList:
    """Clone num_modules number of the module.
    Args:
        module: module to clone
        num_modules: number of modules to create as clones
    """
    return nn.ModuleList([
        copy.deepcopy(module) for _ in range(num_modules)
    ])


def get_device():
    """Get computation device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
