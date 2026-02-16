"""Trainer module for Transformer model training.

This module provides the Trainer class that handles:
- Training loop with gradient updates
- Validation/evaluation
- Checkpoint saving and loading for crash recovery
- Model saving after training completion

Architecture:
    Trainer takes a model and data loader, managing the training lifecycle.
    Separation of concerns:
    - model.py: Model architecture only
    - trainer.py: Training logic, checkpointing, saving/loading
    - app.py: Inference/application layer

Directory Structure:
    {result_dir}/{model_name}/
    ├── snapshots/    # Training checkpoints (deleted after training)
    │   └── snapshot_epoch_0005_step_001000_20240115_143052.pt
    ├── models/       # Completed models
    │   └── model_20240115_150030.pt
    ├── runs/         # TensorBoard event files
    │   └── events.out.tfevents...
    └── logs/         # Training logs

Usage:
    from scratch.trainer import Trainer
    from scratch.model import Transformer

    model = Transformer()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=torch.nn.NLLLoss(),
        model_name="my_translator"
    )

    # Train
    trainer.train(train_loader, val_loader, num_epochs=10)

    # Resume from checkpoint
    trainer.load_snapshot("snapshot_epoch_0005_step_001000_20240115_143052.pt")
    trainer.train(train_loader, val_loader, num_epochs=10)
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    List, Dict, Any, Optional
)

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.utility import (
    ensure_directory_exists,
    build_snapshot_filename,
    build_model_filename,
    resolve_file_path,
    find_latest_file,
    delete_files_by_pattern,
    cleanup_old_files,
)
from training.trainer_callback import (
    CallbackList,
    TrainerCallback
)
from training.ema_loss_monitor import EMALossMonitor
from model.constant import LABEL_IGNORE_VALUE


@dataclass
class TrainerConfig:
    """Configuration for Trainer.

    Note: Early stopping and gradient monitoring are now handled by callbacks.
    See trainer_early_stopping.py and trainer_gradient_monitor.py.

    Attributes:
        model_name: Name for this training run, used as root directory.
        result_dir: Root directory for all output files (default: "result").
        gradient_clip: Maximum gradient norm for clipping (None to disable).
        log_interval: Log training progress every N steps.
        snapshot_interval: Save snapshot every N steps (0 to disable step snapshots).
        snapshot_per_epoch: Save snapshot at the end of each epoch.
        keep_last_n_snapshots: Number of recent snapshots to keep (0 to keep all).
        delete_snapshots_after_training: Delete all snapshots after training completes.
    """
    model_name: str = "transformer"
    result_dir: str = "result"
    gradient_clip: Optional[float] = 1.0
    log_interval: int = 100
    snapshot_interval: int = 0
    snapshot_per_epoch: bool = True
    keep_last_n_snapshots: int = 5
    delete_snapshots_after_training: bool = True
    max_steps: Optional[int] = None  # Stop training after N steps (None = no limit)
    sanity_check_interval: int = 0  # Run weight/gradient validation every N steps (0 to disable)

    # Scheduler stepping policy
    # False = step once per epoch (default).
    # True = step every batch (required for step-based warmup schedules).
    step_scheduler_per_batch: bool = False

    # EMA (Exponential Moving Average) loss tracking
    # Provides smoothed loss visualization alongside raw step loss
    enable_ema_loss: bool = True  # Toggle EMA tracking (negligible overhead)
    ema_alpha: float = 0.99  # EMA decay factor: higher = more smoothing (0.95-0.999)


# ================================================================================
# Seq2Seq Trainer e.g, Language Translation with Teacher Forcing
# ================================================================================
class Trainer:
    """Trainer class for model training with checkpoint management.

    Handles the training loop, validation, checkpointing, and model saving.
    Designed for separation of concerns - model.py handles architecture,
    trainer.py handles training lifecycle.

    Attributes:
        model: The neural network model to train.
        optimizer: Optimizer for gradient updates.
        criterion: Loss function.
        config: Training configuration.
        device: Device to run training on (cuda/cpu).
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            config: Optional[TrainerConfig] = None,
            device: Optional[str] = None,
            scheduler: Optional[Any] = None,
            callbacks: Optional[List['TrainerCallback']] = None
    ):
        """Initialize the Trainer.

        Args:
            model: Neural network model to train.
            optimizer: Optimizer for parameter updates.
            criterion: Loss function for training.
            config: Training configuration (uses defaults if None).
            device: Device string like "cuda" or "cpu" (auto-detect if None).
            scheduler: Optional learning rate scheduler.
            callbacks: List of TrainerCallback instances for extensibility.
        """
        # Core training components
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config or TrainerConfig()

        # Training parameters
        self.current_epoch: int = -1
        self.current_step: int = -1
        self.global_step: int = -1
        self.best_val_loss: float = float("inf")
        self.training_history: List[Dict[str, Any]] = []

        # Auxiliary components
        self.writer: SummaryWriter

        # Setup device and callbacks
        self.device: torch.device = self._setup_device(device)
        self.callbacks: CallbackList = CallbackList(callbacks or [])

        # Initialize directory structure and training state
        self._setup_directories()
        self._initialize_training_state()

        # Setup model and monitoring systems
        self._setup_model_and_logging()
        self._initialize_ema_monitor()

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup and return the compute device.

        Args:
            device: Device string like "cuda" or "cpu" (auto-detect if None).

        Returns:
            torch.device object for training.
        """
        return torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _setup_model_and_logging(self) -> None:
        """Move model to device and initialize TensorBoard logging."""
        self.model.to(self.device)
        self.writer: SummaryWriter = SummaryWriter(log_dir=str(self.model_root_dir / "runs"))

    def _setup_directories(self) -> None:
        """Create directory structure for snapshots, models, and logs."""
        base = Path(self.config.result_dir)
        self.model_root_dir = base / self.config.model_name
        self.snapshots_dir = self.model_root_dir / "snapshots"
        self.models_dir = self.model_root_dir / "models"
        self.logs_dir = self.model_root_dir / "logs"

    def _initialize_training_state(self) -> None:
        """Initialize training state tracking variables."""
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.training_history: List[Dict[str, Any]] = []

    def _initialize_ema_monitor(self) -> None:
        """Initialize EMA loss monitoring system if enabled in config.

        The EMA monitor smooths noisy step-by-step loss values to reveal
        underlying learning trends.
        """
        self.ema_monitor: Optional[EMALossMonitor] = None

        if not self.config.enable_ema_loss:
            return

        # Create EMA monitor with configured alpha
        self.ema_monitor = EMALossMonitor(alpha=self.config.ema_alpha)

        print(
            f"EMA loss monitoring enabled "
            f"(alpha={self.config.ema_alpha})"
        )

    # ================================================================================
    # Training Methods
    # ================================================================================
    def train(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            num_epochs: int = 10,
            start_epoch: int = 0
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (optional).
            num_epochs: Total number of epochs to train.
            start_epoch: Epoch to start from (for resuming training).

        Returns:
            Dictionary with training history and final metrics.
        """
        self._initialize_training_loop(start_epoch, num_epochs)

        try:
            self._run_epoch_loop(train_loader, val_loader, start_epoch, num_epochs)
            return self._finalize_training()
        finally:
            self._close_tensorboard_writer()

    def _initialize_training_loop(
            self,
            start_epoch: int,
            num_epochs: int
    ) -> None:
        """Initialize training loop state and notify callbacks.

        Args:
            start_epoch: Epoch to start from (for resuming training).
            num_epochs: Total number of epochs to train.
        """
        self.current_epoch = start_epoch
        print(f"Training on {self.device} for {num_epochs} epochs")
        self.callbacks.on_train_start(self)

    def _run_epoch_loop(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            start_epoch: int,
            num_epochs: int
    ) -> None:
        """Execute the main training loop across epochs.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (optional).
            start_epoch: Epoch to start from (for resuming training).
            num_epochs: Total number of epochs to train.
        """
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Notify callbacks that epoch is starting
            self.callbacks.on_epoch_start(self, epoch)

            # Train for one epoch
            train_loss = self._train_one_epoch(
                train_loader=train_loader,
                epoch=epoch
            )

            # Check if global step limit was reached during training
            if self._should_stop_for_max_steps():
                break

            # Run post-epoch operations (validation, logging, checkpointing)
            should_stop = self._post_epoch_operations(
                train_loader=train_loader,
                val_loader=val_loader,
                epoch=epoch,
                train_loss=train_loss
            )

            # Check if training should stop early
            if should_stop:
                break

    def _post_epoch_operations(
            self,
            train_loader: DataLoader,   # pylint: disable=unused-argument
            val_loader: Optional[DataLoader],
            epoch: int,
            train_loss: float
    ) -> bool:
        """Execute post-epoch operations and return whether to stop training.
        Note:
            Currently the train_loader argument is not used but keep it in the
            signature for potential future use in callbacks.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (optional).
            epoch: Current epoch number.
            train_loss: Average training loss for this epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        # Run validation if validation loader is provided
        val_loss = self._validate(val_loader) if val_loader else None

        # Log metrics and update learning rate scheduler
        self._log_epoch_summary(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss
        )
        self._update_scheduler()

        # Notify callbacks that epoch has ended
        self.callbacks.on_epoch_end(self, epoch, train_loss, val_loss)

        # Save checkpoint after callbacks have updated their state
        self._save_epoch_checkpoint(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss
        )

        # Check if callbacks want to stop training early
        return self.callbacks.should_stop_training(self)

    def _should_stop_for_max_steps(self) -> bool:
        """Check if training should stop due to reaching max_steps limit.

        Returns:
            True if max_steps is configured and has been reached.
        """
        if self.config.max_steps is None:
            return False
        return self.global_step >= self.config.max_steps

    def _close_tensorboard_writer(self) -> None:
        """Safely close the TensorBoard writer, handling errors gracefully."""
        if not hasattr(self, 'writer') or self.writer is None:
            return

        try:
            self.writer.close()
        except Exception as e:
            print(f"Warning: Failed to close TensorBoard writer: {e}")

    def _train_one_epoch(
            self,
            train_loader: DataLoader,
            epoch: int
    ) -> float:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data.
            epoch: Current epoch number.

        Returns:
            Average training loss for this epoch.
        """
        # Set model to training mode (enables dropout, batch norm updates, etc.)
        self.model.train()

        num_batches = len(train_loader)
        if num_batches == 0:
            print("Warning: Training dataloader is empty, returning 0.0 loss")
            return 0.0

        # Process all batches in this epoch
        total_loss = self._process_epoch_batches(
            train_loader=train_loader,
            epoch=epoch,
            num_batches=num_batches
        )

        return total_loss / num_batches

    def _process_epoch_batches(
            self,
            train_loader: DataLoader,
            epoch: int,
            num_batches: int
    ) -> float:
        """Process all batches in an epoch and return total loss.

        Args:
            train_loader: DataLoader for training data.
            epoch: Current epoch number.
            num_batches: Total number of batches in this epoch.

        Returns:
            Sum of losses across all processed batches.
        """
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            # Process one batch and update counters
            loss = self._process_one_batch(
                batch=batch,
                step=step,
                epoch=epoch,
                num_batches=num_batches
            )
            total_loss += loss

            # Check if global step limit has been reached
            if self._should_stop_for_max_steps():
                print(
                    f"\nReached max_steps={self.config.max_steps}. "
                    "Stopping training."
                )
                # Return average of batches processed so far
                return total_loss / (step + 1) * num_batches

        return total_loss

    def _process_one_batch(
            self,
            batch: Dict[str, Tensor],
            step: int,
            epoch: int,
            num_batches: int
    ) -> float:
        """Process one batch and perform all related operations.

        Args:
            batch: Dictionary containing batch data.
            step: Current step number within epoch.
            epoch: Current epoch number.
            num_batches: Total number of batches in this epoch.

        Returns:
            Loss value for this batch.
        """
        # Notify callbacks that batch processing is starting
        self.callbacks.on_batch_start(self, step)

        # Execute forward pass, backward pass, and optimizer step
        loss = self._train_one_step(batch)

        # Update step counters
        self.current_step = step
        self.global_step += 1

        # Log progress and save checkpoints if configured
        self._log_step_progress(
            epoch=epoch,
            step=step,
            num_batches=num_batches,
            loss=loss
        )
        self._save_step_checkpoint(epoch=epoch, step=step)

        # Notify callbacks that batch processing has ended
        self.callbacks.on_batch_end(self, step, loss)

        return loss

    def _train_one_step(self, batch: Dict[str, Tensor]) -> float:
        r"""Execute one training step (forward, backward, update).

        The loss is computed using Negative Log-Likelihood:
        $$ \\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log P(y_i | x_i) $$

        Args:
            batch: Dictionary containing 'source_ids', 'target_ids', 'source_pad_mask',
                and optionally 'target_pad_mask'.

        Returns:
            Loss value for this step.
        """
        # --------------------------------------------------------------------------------
        # For EN to ES translation task:
        # source_ids: English sequences (encoder input)
        # target_ids: Spanish sequences with <BOS> prepended and <EOS> appended
        #
        # decoder_input = target_ids[:, :-1]:
        # Spanish sequence WITHOUT the last token used as decoder input
        # decoder_target = target_ids[:, 1:]:
        # Spanish sequence WITHOUT the first token used for loss calculation
        #
        # Example:
        # Original texts:
        # - English: "The cat sat"
        # - Spanish: "El gato sentó"
        #
        # After tokenization:
        # source_ids = [101, 2003, 4860, 102]               # "The cat sat <EOS>"
        # target_ids = [50256, 2573, 5721, 3242, 50256]     # "<BOS> El gato sentó <EOS>"
        #
        # In trainer (teacher forcing):
        # decoder_input  = [50256, 2573, 5721, 3242]        # "<BOS> El gato sentó"
        # decoder_target = [2573, 5721, 3242, 50256]        # "El gato sentó <EOS>"
        # --------------------------------------------------------------------------------
        source_ids = batch["source_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)

        # Extract optional source padding mask for attention
        source_pad_mask = batch.get("source_pad_mask")
        if source_pad_mask is not None:
            source_pad_mask = source_pad_mask.to(self.device)

        # Extract optional target padding mask for loss masking
        target_pad_mask = batch.get("target_pad_mask")
        if target_pad_mask is not None:
            target_pad_mask = target_pad_mask.to(self.device)

        # --------------------------------------------------------------------------------
        # Shift target for teacher forcing
        # --------------------------------------------------------------------------------
        # decoder_input = target_ids[:, :-1]  <- uses clean token IDs for embedding
        # decoder_target = target_ids[:, 1:]  <- will be masked for loss calculation
        #
        # CRITICAL: target_ids must contain valid token IDs (not LABEL_IGNORE_VALUE)
        # for embedding lookup. Masking is applied to decoder_target only.
        # --------------------------------------------------------------------------------
        decoder_input = target_ids[:, :-1]  # Source tokens to predict next. Shape: (B, T-1)
        decoder_target = target_ids[:, 1:]  # Next tokens as target labels. Shape: (B, T-1)

        # --------------------------------------------------------------------------------
        # Apply target padding mask to decoder_target for loss calculation
        # --------------------------------------------------------------------------------
        # Mask targets labels by IGNORE_VALUE when their values are pad token id.
        # Then, the loss function ignores those labels from loss calculation.
        #
        # CRITICAL:
        # Do NOT use pad_token_id as the target mask value for GPT-like tokenizers.
        # They usually do not define a dedicated PAD token by default.
        # Hence, what people will do is pad_token_id = eos_token_id = eot_token_id.
        # EOS is now the PAD token for the model to ignore in loss calculation.
        #
        # The decoder must learn P(EOS | last_token_in_sequence).
        # This is how the decoder model learns the human notion of "End of Sequence",
        # or Semantic Boundaries in general.
        #
        # EOS is the token that teaches the model where/how a sequence ends.
        # In corpora for GPT-like models, the EOS token is a semantic boundary.
        # Without learning it, the model has no understanding of "End",
        # or the concept of "Sequence" or "Boundary". Decoder may keep generating or
        # only stop at max length as it does not know "End".
        #
        # Another potential side effect:
        # If padding masks or positional handling are incorrect, the same sentence
        # padded to length 64 vs 128 may produce different attention distributions
        # and different outputs, even when the real tokens are identical.
        # --------------------------------------------------------------------------------
        # NOTE: Padding Mask at Attention
        # Padding must be handled at the attention level (not just in the loss) as well.
        # If PAD tokens are included as keys/values (K,V) and not masked, queries (Q) from
        # real tokens can attend to PAD positions. This wastes attention mass and can
        # contaminate real-token representations as attention-weighted sum includes PAD.
        #
        # To address it, apply a padding mask which is to set attention logits to -inf
        # for PAD token positions.
        # --------------------------------------------------------------------------------
        # Prepare masks for both loss calculation and attention
        decoder_input_pad_mask = None  # For decoder self-attention
        if target_pad_mask is not None:
            # Shift target_pad_mask to match decoder_input for attention masking
            # decoder_input = target_ids[:, :-1], so mask should also drop last position
            decoder_input_pad_mask = target_pad_mask[:, :-1]

            # Shift target_pad_mask to match decoder_target for loss masking
            # decoder_target = target_ids[:, 1:], so mask should also drop first position
            target_pad_mask_shifted = target_pad_mask[:, 1:]
            decoder_target = decoder_target.masked_fill(
                target_pad_mask_shifted,
                LABEL_IGNORE_VALUE
            )

        self.optimizer.zero_grad()
        log_probabilities = self.model(
            x=source_ids,
            y=decoder_input,
            source_pad_mask=source_pad_mask,
            target_pad_mask=decoder_input_pad_mask
        )

        loss = self._compute_loss(log_probabilities, decoder_target)

        # Callback: on_forward_end
        self.callbacks.on_forward_end(self, loss)

        loss.backward()

        # Callback: on_backward_end (gradients computed, not yet clipped)
        self.callbacks.on_backward_end(self)

        self._clip_gradients()
        self.optimizer.step()

        # Step scheduler per batch if configured (e.g., warmup schedules)
        self._step_scheduler_if_configured()

        # Callback: on_step_end
        self.callbacks.on_step_end(self)

        return loss.item()

    def _step_scheduler_if_configured(self) -> None:
        """Step the learning rate scheduler if per-batch stepping is enabled.

        This method is extracted separately to ensure tests exercise the real
        production scheduler stepping logic even when _train_one_step is patched.

        Called by _train_one_step after optimizer.step() when
        config.step_scheduler_per_batch=True (required for warmup schedules).
        """
        if self.scheduler is not None and self.config.step_scheduler_per_batch:
            self.scheduler.step()

    def _compute_loss(self, log_probabilities: Tensor, targets: Tensor) -> Tensor:
        """Compute loss between predictions and targets.

        Args:
            log_probabilities: Log probabilities of shape (B, T, V).
            targets: Target token indices of shape (B, T).

        Returns:
            Scalar loss tensor.
        """
        # Flatten for loss computation: (B, T, V) -> (B*T, V), (B, T) -> (B*T)
        log_probs_flat = log_probabilities.reshape(-1, log_probabilities.size(-1))
        targets_flat = targets.reshape(-1)
        return self.criterion(log_probs_flat, targets_flat)

    def _clip_gradients(self) -> None:
        """Clip gradients to prevent exploding gradients."""
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

    def _update_scheduler(self) -> None:
        """Update learning rate scheduler if present."""
        if self.scheduler is not None and not self.config.step_scheduler_per_batch:
            self.scheduler.step()

    # ================================================================================
    # Validation Methods
    # ================================================================================
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Evaluate model on validation set.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            # See translation_collate_fn() in loader_translation.py for details
            # about how a batch is constructed and what keys it contains.
            source_ids = batch["source_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            # Shift target for teacher forcing in validation as well
            decoder_input = target_ids[:, :-1]  # Prev tokens of shape (B, T-1)
            decoder_target = target_ids[:, 1:]  # Next tokens of Shape (B, T-1)

            # Extract optional source padding mask for attention
            source_pad_mask = batch.get("source_pad_mask")
            if source_pad_mask is not None:
                source_pad_mask = source_pad_mask.to(self.device)

            # Extract optional target padding mask for loss masking
            target_pad_mask = batch.get("target_pad_mask")
            if target_pad_mask is not None:
                target_pad_mask = target_pad_mask.to(self.device)

            # Prepare masks for both loss calculation and attention
            decoder_input_pad_mask = None  # For decoder self-attention
            if target_pad_mask is not None:
                # Shift target_pad_mask to match decoder_input for attention masking
                decoder_input_pad_mask = target_pad_mask[:, :-1]

                # Shift target_pad_mask to match decoder_target for loss masking
                target_pad_mask_shifted = target_pad_mask[:, 1:]
                decoder_target = decoder_target.masked_fill(
                    target_pad_mask_shifted,
                    LABEL_IGNORE_VALUE
                )

            log_probabilities = self.model(
                x=source_ids,
                y=decoder_input,
                source_pad_mask=source_pad_mask,
                target_pad_mask=decoder_input_pad_mask
            )
            loss = self._compute_loss(log_probabilities, decoder_target)
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            print("Warning: Validation dataloader is empty, returning 0.0 loss")
            return 0.0

        return total_loss / num_batches

    # ================================================================================
    # Logging Methods
    # ================================================================================
    def _check_weights_valid(self) -> tuple[bool, list[str]]:
        """Check if any model weights contain NaN or Inf values.

        Returns:
            Tuple of (is_valid, list_of_invalid_params)
        """
        invalid_params = []

        for name, param in self.model.named_parameters():
            if torch.isnan(param.data).any():
                invalid_params.append(f"{name} contains NaN")
            if torch.isinf(param.data).any():
                invalid_params.append(f"{name} contains Inf")

        return len(invalid_params) == 0, invalid_params

    def _check_gradients_valid(self) -> tuple[bool, list[str]]:
        """Check if any gradients contain NaN or Inf values.

        Returns:
            Tuple of (is_valid, list_of_invalid_grads)
        """
        invalid_grads = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    invalid_grads.append(f"{name}.grad contains NaN")
                if torch.isinf(param.grad).any():
                    invalid_grads.append(f"{name}.grad contains Inf")

        return len(invalid_grads) == 0, invalid_grads

    def _log_step_progress(
            self,
            epoch: int,
            step: int,
            num_batches: int,
            loss: float
    ) -> None:
        """Log training progress at step level."""
        self.writer.add_scalar("train/step_loss", loss, self.global_step)

        # Update and log EMA loss if enabled
        ema_loss = None
        if self.ema_monitor is not None:
            ema_loss = self.ema_monitor.update(loss)
            self.writer.add_scalar("train/loss_ema", ema_loss, self.global_step)

        # Periodic weight and gradient validation (sanity checks)
        if (
                self.config.sanity_check_interval > 0 and \
                self.global_step % self.config.sanity_check_interval == 0 and \
                self.global_step > 0
        ):
            is_valid_weights, invalid_params = self._check_weights_valid()
            is_valid_grads, invalid_grads = self._check_gradients_valid()

            if not is_valid_weights:
                print(f"  WARNING at step {self.global_step}: Invalid weights detected")
                for msg in invalid_params:
                    print(f"    - {msg}")

            if not is_valid_grads:
                print(f"  WARNING at step {self.global_step}: Invalid gradients detected")
                for msg in invalid_grads:
                    print(f"    - {msg}")

            if is_valid_weights and is_valid_grads:
                print(f"  ✓ Validation at step {self.global_step}: All weights and gradients valid")

        if (step + 1) % self.config.log_interval == 0 or step == 0:
            msg = f"  Epoch {epoch} | Step {step + 1}/{num_batches} | Loss: {loss:.4f}"
            if ema_loss is not None:
                msg += f" | EMA: {ema_loss:.4f}"
            print(msg)

    def _log_epoch_summary(
            self,
            epoch: int,
            train_loss: float,
            val_loss: Optional[float]
    ) -> None:
        """Log summary at end of epoch."""
        msg = f"Epoch {epoch} | Train Loss: {train_loss:.4f}"
        self.writer.add_scalar("train/epoch_loss", train_loss, epoch)
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
            self.writer.add_scalar("val/epoch_loss", val_loss, epoch)
        print(msg)

        self.training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        self._log_weight_stats(epoch)

    def _log_weight_stats(self, epoch: int) -> None:
        """Check for NaN/Inf in weights (aggregate only, no per-param histograms)."""
        nan_param_count = 0
        inf_param_count = 0

        for name, param in self.model.named_parameters():   # pylint: disable=unused-variable
            # Check for NaN/Inf in weights
            if torch.isnan(param.data).any():
                nan_param_count += 1
            if torch.isinf(param.data).any():
                inf_param_count += 1

        # Log aggregate validity statistics only
        self.writer.add_scalar("debug/nan_param_count", nan_param_count, epoch)
        self.writer.add_scalar("debug/inf_param_count", inf_param_count, epoch)

        if nan_param_count > 0 or inf_param_count > 0:
            print(f"  🔴 Epoch {epoch}: {nan_param_count} params with NaN, {inf_param_count} with Inf")

    # ================================================================================
    # Checkpoint Methods
    # ================================================================================
    def _save_step_checkpoint(self, epoch: int, step: int) -> None:
        """Save checkpoint at step interval if configured."""
        interval = self.config.snapshot_interval
        if interval > 0 and (step + 1) % interval == 0:
            self.save_snapshot(epoch, step)
            if self.config.keep_last_n_snapshots > 0:
                self.cleanup_old_snapshots(self.config.keep_last_n_snapshots)

    def _save_epoch_checkpoint(
            self,
            epoch: int,
            train_loss: float,
            val_loss: Optional[float] = None
    ) -> None:
        """Save checkpoint at end of epoch if configured.

        Args:
            epoch: Current epoch number.
            train_loss: Average training loss for this epoch.
            val_loss: Average validation loss for this epoch, or None if validation
                was not performed.
        """
        if not self.config.snapshot_per_epoch:
            return

        # --------------------------------------------------------------------------------
        # Loss Selection for Checkpoint Tracking
        #
        # Use val_loss if available, otherwise fall back to train_loss.
        # CRITICAL: Use explicit None check, not 'or' operator.
        # Bug: 'val_loss or train_loss' treats 0.0 as falsy, incorrectly falling back
        # to train_loss even when validation succeeded with perfect loss.
        # --------------------------------------------------------------------------------
        current_loss = val_loss if val_loss is not None else train_loss

        # Save checkpoint with current loss
        self.save_snapshot(epoch=epoch, step=0, loss=current_loss)

        # Update best model tracking
        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss

        # Cleanup old snapshots if retention limit is configured
        if self.config.keep_last_n_snapshots > 0:
            self.cleanup_old_snapshots(
                keep_last_n=self.config.keep_last_n_snapshots
            )

    def save_snapshot(
            self,
            epoch: int,
            step: int = 0,
            loss: Optional[float] = None,
            extra_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save training snapshot for crash recovery.

        Args:
            epoch: Current epoch number.
            step: Current step within epoch.
            loss: Current loss value (optional).
            extra_state: Additional state to save (optional).

        Returns:
            Path to saved snapshot file.
        """
        ensure_directory_exists(self.snapshots_dir)

        # Collect and merge callback state into extra_state
        extra_state = self._merge_callback_state(extra_state, epoch, step)

        # Build checkpoint dictionary and determine save path
        checkpoint = self._build_checkpoint_dict(epoch, step, loss, extra_state)
        filepath = self.snapshots_dir / build_snapshot_filename(epoch, step)

        # Save checkpoint to disk
        torch.save(checkpoint, filepath)
        print(f"Snapshot saved: {filepath}")
        return filepath

    def _merge_callback_state(
            self,
            extra_state: Optional[Dict[str, Any]],
            epoch: int,
            step: int
    ) -> Dict[str, Any]:
        """Merge callback state into extra_state dictionary.

        Args:
            extra_state: Additional state to save (optional).
            epoch: Current epoch number.
            step: Current step within epoch.

        Returns:
            Merged extra_state dictionary with callback state.
        """
        # Initialize extra_state if None
        if extra_state is None:
            extra_state = {}

        # Collect callback state and merge if present
        callback_state = self.callbacks.on_snapshot_save(self, epoch, step)
        if callback_state:
            extra_state['callback_state'] = callback_state

        return extra_state

    def _build_checkpoint_dict(
            self,
            epoch: int,
            step: int,
            loss: Optional[float],
            extra_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build dictionary containing all checkpoint data."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "global_step": self.global_step,
            "loss": loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.ema_monitor is not None:
            checkpoint["ema_state"] = self.ema_monitor.state_dict()
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state
        return checkpoint

    def load_snapshot(
            self,
            filename: str,
            map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load training snapshot to resume training.

        Args:
            filename: Snapshot filename or full path.
            map_location: Device to load tensors to (e.g., "cpu", "cuda:0").

        Returns:
            Checkpoint dictionary with epoch, step, loss, etc.
        """
        filepath = resolve_file_path(filename, self.snapshots_dir)
        checkpoint = torch.load(filepath, map_location=map_location)

        self._restore_from_checkpoint(checkpoint)

        # Callback: restore callback state
        self.callbacks.on_snapshot_load(self, checkpoint)

        print(f"Snapshot loaded: {filepath} (epoch {self.current_epoch})")
        return checkpoint

    def _restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore all state from checkpoint dictionary."""
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.current_step = checkpoint.get("step", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.ema_monitor is not None and "ema_state" in checkpoint:
            self.ema_monitor.load_state_dict(checkpoint["ema_state"])

    def find_latest_snapshot(self) -> Optional[Path]:
        """Find the most recent snapshot file.

        Returns:
            Path to latest snapshot, or None if none exist.
        """
        return find_latest_file(self.snapshots_dir, "snapshot_*.pt")

    def cleanup_old_snapshots(self, keep_last_n: int = 5) -> int:
        """Remove old snapshots, keeping only the most recent ones.

        Args:
            keep_last_n: Number of recent snapshots to keep.

        Returns:
            Number of snapshots deleted.
        """
        deleted = cleanup_old_files(self.snapshots_dir, "snapshot_*.pt", keep_last_n)
        if deleted > 0:
            print(f"Cleaned up {deleted} old snapshots")
        return deleted

    # ================================================================================
    # Model Saving Methods
    # ================================================================================
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training: save model and cleanup snapshots."""
        model_path = self.save_model()

        if self.config.delete_snapshots_after_training:
            self._delete_all_snapshots()

        # Close writer if not already closed (normal completion path)
        if hasattr(self, 'writer') and self.writer is not None:
            try:
                self.writer.close()
                self.writer = None
            except Exception:
                pass  # Already closed or error, ignore

        result = {
            "model_path": model_path,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "total_epochs": self.current_epoch + 1,
        }

        # Callback: on_train_end
        self.callbacks.on_train_end(self, result)

        return result

    def save_model(self, filename: Optional[str] = None) -> Path:
        """Save completed model for inference.

        Args:
            filename: Custom filename without .pt extension (optional).

        Returns:
            Path to saved model file.
        """
        ensure_directory_exists(self.models_dir)

        filename = build_model_filename(filename)
        filepath = self.models_dir / filename

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "model_name": self.config.model_name,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "timestamp": datetime.now().isoformat(),
        }

        torch.save(save_dict, filepath)
        print(f"Model saved: {filepath}")
        return filepath

    def load_model(
            self,
            filename: str,
            map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load a saved model for inference or further training.

        Args:
            filename: Model filename or full path.
            map_location: Device to load tensors to.

        Returns:
            Dictionary containing model metadata.
        """
        filepath = resolve_file_path(filename, self.models_dir)
        save_dict = torch.load(filepath, map_location=map_location)

        self.model.load_state_dict(save_dict["model_state_dict"])
        self.best_val_loss = save_dict.get("best_val_loss", float("inf"))
        self.training_history = save_dict.get("training_history", [])

        print(f"Model loaded: {filepath}")
        return save_dict

    def _delete_all_snapshots(self) -> int:
        """Delete all snapshot files after training completion."""
        deleted = delete_files_by_pattern(self.snapshots_dir, "snapshot_*.pt")
        if deleted > 0:
            print(f"Deleted {deleted} training snapshots")
        return deleted


# ================================================================================
# Decoder Only Model Trainer (e.g. GPT-style)
# ================================================================================
class LanguageModelTrainer(Trainer):
    """Trainer subclass for decoder-only Language Models (GPT-style).

    Unlike Trainer which expects encoder-decoder models with source_ids and
    target_ids, LanguageModelTrainer handles language models where:
    - Batch is a tuple (input_ids, target_ids) where target is input shifted by 1
    - Model(x) takes only input, no source/memory

    Usage:
        from scratch.lm import LanguageModel
        from scratch.trainer import LanguageModelTrainer, TrainerConfig

        model = LanguageModel(vocab_size=50257)
        trainer = LanguageModelTrainer(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            criterion=torch.nn.NLLLoss(),
            config=TrainerConfig(model_name="gpt_wikitext")
        )
        trainer.train(train_loader, val_loader, num_epochs=10)
    """

    def _train_one_step(self, batch: Tensor) -> float:
        r"""Execute one training step for language model.

        The loss is computed using Negative Log-Likelihood:
        $$ \\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log P(x_t | x_{<t}) $$

        Args:
            batch: Tuple of (input_ids, target_ids) tensors.

        Returns:
            Loss value for this step.
        """
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # DEBUG: Sanity check every 5000 steps
        if self.global_step % 5000 == 0 and self.global_step > 0:
            print("\n" + "="*70)
            print(f"SANITY CHECK AT STEP {self.global_step}")
            print("="*70)
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"Input shape: {input_ids.shape}")
            print(f"Target shape: {target_ids.shape}")
            non_ignored = (target_ids != -100).sum().item()
            total = target_ids.numel()
            print(f"Non-ignored labels: {non_ignored}/{total} ({100*non_ignored/total:.1f}%)")
            print(f"Sequence length T: {input_ids.shape[1]}")
            print("="*70 + "\n")

        self.optimizer.zero_grad()
        log_probabilities: Tensor = self.model(input_ids)

        loss: Tensor = self._compute_loss(log_probabilities, target_ids)

        # Callback: on_forward_end
        self.callbacks.on_forward_end(self, loss)

        # Back-propagate gradients to populates param.grad for each parameter.
        # No weight change yet.
        loss.backward()

        # Callback: on_backward_end (gradients computed, not yet clipped)
        self.callbacks.on_backward_end(self)

        self._clip_gradients()

        # Update weights using the gradients in param.grad according to the optimizer rule.
        self.optimizer.step()

        # Step scheduler per batch if configured (e.g., warmup schedules)
        self._step_scheduler_if_configured()

        # Callback: on_step_end
        self.callbacks.on_step_end(self)

        return loss.item()

    @torch.no_grad()
    def _validate(self, val_loader) -> float:
        """Evaluate language model on validation set.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            log_probabilities = self.model(input_ids)
            loss = self._compute_loss(log_probabilities, target_ids)
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            print("Warning: Validation dataloader is empty, returning 0.0 loss")
            return 0.0

        return total_loss / num_batches


if __name__ == "__main__":
    print(__doc__)
