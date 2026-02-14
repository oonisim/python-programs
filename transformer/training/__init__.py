"""Training infrastructure for transformer models.

This module contains the training system:
- trainer.py: Main Trainer and LanguageModelTrainer classes
- trainer_callback.py: Callback system for training hooks
- trainer_early_stopping.py: Early stopping callback
- trainer_gradient_monitor.py: Gradient flow monitoring callback
- train_lm.py: Language model training director
- train_translation.py: Translation training director
- loader.py: Data loader for language modeling
- loader_translation.py: Data loader for translation
- gradient_monitor.py: Gradient flow monitoring utilities
- utility.py: Training utilities (checkpointing, file management)
"""

# Main trainer classes
from training.trainer import Trainer, LanguageModelTrainer, TrainerConfig

# Callback system
from training.trainer_callback import TrainerCallback, CallbackList
from training.trainer_early_stopping import EarlyStoppingCallback
from training.trainer_gradient_monitor import GradientMonitorCallback

# Data loaders
from training.loader import LanguageModelDataLoaderFactory, DataLoaderConfig
from training.loader_translation import TranslationDataLoaderFactory

# Monitoring
from training.gradient_monitor import GradientGainMonitor

# Utilities
from training.utility import (
    ensure_directory_exists,
    build_snapshot_filename,
    build_model_filename,
    resolve_file_path,
    find_latest_file,
    delete_files_by_pattern,
    cleanup_old_files,
)

__all__ = [
    # Trainers
    'Trainer',
    'LanguageModelTrainer',
    'TrainerConfig',
    # Callbacks
    'TrainerCallback',
    'CallbackList',
    'EarlyStoppingCallback',
    'GradientMonitorCallback',
    # Data loaders
    'LanguageModelDataLoaderFactory',
    'TranslationDataLoaderFactory',
    'DataLoaderConfig',
    # Monitoring
    'GradientGainMonitor',
    # Utilities
    'ensure_directory_exists',
    'build_snapshot_filename',
    'build_model_filename',
    'resolve_file_path',
    'find_latest_file',
    'delete_files_by_pattern',
    'cleanup_old_files',
]
