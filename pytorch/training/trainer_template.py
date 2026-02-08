import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module

def train_epoch(
        model: Module,
        data_loader: DataLoader,
        optimizer: Optimizer,
        loss_function: Module,
        device: torch.device
) -> float:
    """
    Executes a single training epoch across the data loader.
    """
    # Set model to training mode (activates Dropout/LayerNorm)
    model.train()
    total_epoch_loss: float = 0.0

    for batch_idx, (source_sequences, target_sequences) in enumerate(data_loader):
        # Move tensors to the computation device (CPU/GPU)
        source_sequences = source_sequences.to(device)
        target_sequences = target_sequences.to(device)

        # 1. Forward Pass
        # For Transformers, the target is often shifted for causal prediction
        predictions = model(source_sequences, target_sequences[:, :-1])

        # 2. Compute Loss
        # Reshape for cross-entropy: (Batch * Time, Vocabulary)
        loss = loss_function(
            predictions.reshape(-1, predictions.shape[-1]),
            target_sequences[:, 1:].reshape(-1)
        )

        # 3. Backward Pass (Gradient Calculation)
        optimizer.zero_grad()  # Reset existing gradients
        loss.backward()        # Backpropagation

        # 4. Optimization Step
        optimizer.step()       # Update weights

        total_epoch_loss += loss.item()

    return total_epoch_loss / len(data_loader)