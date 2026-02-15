"""Module for Transformers utilities.

This module contains:
- Module cloning utilities
- Device selection utilities
- Parameter counting utilities (accounting for weight tying)
- File and directory utilities for checkpoint management
"""
import copy
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn


# ================================================================================
# Module Utilities
# ================================================================================
def clone_module(module: nn.Module, num_modules: int) -> nn.ModuleList:
    """Clone num_modules number of the module.

    Args:
        module: Module to clone.
        num_modules: Number of clones to create.

    Returns:
        ModuleList containing the cloned modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_modules)])


def get_device() -> torch.device:
    """Get the best available computation device (CUDA if available, else CPU).

    Returns:
        torch.device for computation.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_device_consistency(module: torch.nn.Module) -> torch.device:
    """
    Verifies all parameters and buffers are on the same device.

    Returns:
        The common torch.device.
    Raises:
        RuntimeError: If tensors are discovered on multiple devices.
    """
    devices = {param.device for param in module.parameters()}
    devices.update({buffer.device for buffer in module.buffers()})

    if len(devices) > 1:
        raise RuntimeError(
            f"Device Inconsistency Detected in {module.__class__.__name__}: "
            f"Found multiple devices: {devices}. "
            f"Ensure model.to(device) was called correctly."
        )

    return next(iter(devices)) if devices else torch.device("cpu")


# ================================================================================
# Parameter Counting Utilities
# ================================================================================
def count_model_parameters(model: nn.Module) -> dict:
    """Count model parameters accounting for weight tying.

    When weight tying is used (e.g., sharing embedding weights with output projection),
    the standard parameter counting approach will count the same tensor multiple times.
    This function correctly counts each unique parameter tensor only once.

    Args:
        model: PyTorch model (e.g., Transformer or LanguageModel).

    Returns:
        Dictionary containing:
        - 'total_parameters': Total unique parameters (accounts for weight tying)
        - 'tied_parameters': Number of parameters that are shared/tied
        - 'trainable_parameters': Number of trainable unique parameters

    Example:
        >>> model = Transformer(decoder_vocabulary_size=50000, decoder_model_dimension=512)
        >>> stats = count_model_parameters(model)
        >>> print(f"Total parameters: {stats['total_parameters']:,}")
        Total parameters: 65,000,000
        >>> print(f"Tied parameters: {stats['tied_parameters']:,}")
        Tied parameters: 25,600,000

    Note:
        Standard counting (sum(p.numel() for p in model.parameters())) will
        double-count tied weights, giving inflated parameter counts.
    """
    all_params = list(model.parameters())
    unique_params = {id(p): p for p in all_params}

    total_unique = sum(p.numel() for p in unique_params.values())
    total_with_duplicates = sum(p.numel() for p in all_params)

    return {
        'total_parameters': total_unique,
        'tied_parameters': total_with_duplicates - total_unique,
        'trainable_parameters': sum(
            p.numel() for p in unique_params.values() if p.requires_grad
        ),
    }


# ================================================================================
# File and Directory Utilities
# ================================================================================
def ensure_directory_exists(directory: Path) -> None:
    """Create directory and parents if they do not exist.

    Args:
        directory: Path to the directory to create.
    """
    directory.mkdir(parents=True, exist_ok=True)


def generate_timestamp() -> str:
    """Generate timestamp string in YYYYMMDD_HHMMSS format.

    Returns:
        Formatted timestamp string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_snapshot_filename(epoch: int, step: int) -> str:
    """Build filename for a training snapshot with timestamp.

    Args:
        epoch: Current training epoch number.
        step: Current step within the epoch.

    Returns:
        Filename in format: snapshot_epoch_{NNNN}_step_{NNNNNN}_{YYYYMMDD}_{HHMMSS}.pt
    """
    timestamp = generate_timestamp()
    return f"snapshot_epoch_{epoch:04d}_step_{step:06d}_{timestamp}.pt"


def build_model_filename(custom_name: Optional[str] = None) -> str:
    """Build filename for a completed model with timestamp.

    Args:
        custom_name: Optional custom name (without .pt extension).

    Returns:
        Filename in format: model_{YYYYMMDD}_{HHMMSS}.pt or {custom_name}.pt
    """
    if custom_name is not None:
        return f"{custom_name}.pt" if not custom_name.endswith(".pt") else custom_name
    timestamp = generate_timestamp()
    return f"model_{timestamp}.pt"


def resolve_file_path(filename: str, default_directory: Path) -> Path:
    """Resolve filename to full path, checking default directory if needed.

    Args:
        filename: Filename or full path to resolve.
        default_directory: Directory to check if filename is not a full path.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If file does not exist in either location.
    """
    filepath = Path(filename)
    if filepath.exists():
        return filepath

    filepath = default_directory / filename
    if filepath.exists():
        return filepath

    raise FileNotFoundError(f"File not found: {filename} (checked {default_directory})")


def find_latest_file(directory: Path, pattern: str = "*.pt") -> Optional[Path]:
    """Find the most recently modified file matching pattern in directory.

    Args:
        directory: Directory to search in.
        pattern: Glob pattern to match files (default: "*.pt").

    Returns:
        Path to the most recent file, or None if no files found.
    """
    if not directory.exists():
        return None

    matching_files = list(directory.glob(pattern))
    if not matching_files:
        return None

    return max(matching_files, key=lambda path: path.stat().st_mtime)


def delete_files_by_pattern(directory: Path, pattern: str = "snapshot_*.pt") -> int:
    """Delete all files matching pattern in directory.

    Args:
        directory: Directory containing files to delete.
        pattern: Glob pattern to match files (default: "snapshot_*.pt").

    Returns:
        Number of files deleted.
    """
    if not directory.exists():
        return 0

    matching_files = list(directory.glob(pattern))
    for file_path in matching_files:
        file_path.unlink()

    return len(matching_files)


def cleanup_old_files(directory: Path, pattern: str, keep_last_n: int = 5) -> int:
    """Remove old files matching pattern, keeping only the most recent ones.

    Args:
        directory: Directory containing files.
        pattern: Glob pattern to match files.
        keep_last_n: Number of recent files to keep (default: 5).

    Returns:
        Number of files deleted.
    """
    if not directory.exists():
        return 0

    matching_files = list(directory.glob(pattern))
    if len(matching_files) <= keep_last_n:
        return 0

    # Sort by modification time (newest first) and delete older ones
    matching_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    deleted_count = 0

    for old_file in matching_files[keep_last_n:]:
        old_file.unlink()
        deleted_count += 1

    return deleted_count
