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

Note: This module does not eagerly import heavy dependencies.
Import explicitly what you need, e.g.:
    from training.trainer import Trainer
    from training.loader import LanguageModelDataLoaderFactory
"""

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
