"""Integration tests with real Transformer models.

Tests callback integration with actual Transformer/LanguageModel architectures,
not just dummy models.
"""
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import Trainer, TrainerConfig
from training.trainer_callback import TrainerCallback
from training.trainer_gradient_monitor import GradientMonitorCallback
from training.trainer_early_stopping import EarlyStoppingCallback


def test_gradient_monitor_with_decoder_layers():
    """Test GradientMonitorCallback with model.decoder.layers structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Gradient Monitor with Decoder Layers")
        print("=" * 70)

        # Create a model with decoder.layers structure (like LanguageModel)
        class MockDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.TransformerDecoderLayer(d_model=64, nhead=4, dim_feedforward=128)
                    for _ in range(3)
                ])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x, x)  # Self-attention only
                return x

        class MockLanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.decoder = MockDecoder()
                self.output = nn.Linear(64, 100)

            def forward(self, x, y=None):
                # x shape: (batch, seq_len)
                embedded = self.embedding(x)
                decoded = self.decoder(embedded)
                logits = self.output(decoded)
                return torch.log_softmax(logits, dim=-1)

        model = MockLanguageModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.NLLLoss()

        # Create gradient monitor callback
        grad_monitor = GradientMonitorCallback(
            monitor_at_snapshots=False,
            monitor_interval=1,  # Monitor every step
            monitor_at_epochs=False,
            norm_type='l2'
        )

        config = TrainerConfig(
            model_name="test_decoder_layers",
            result_dir=tmpdir,
            log_interval=100,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            callbacks=[grad_monitor]
        )

        # Create sequence data
        x = torch.randint(0, 100, (16, 10))  # (batch, seq_len)
        y = torch.randint(0, 100, (16, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        # Monkey-patch to work with our data format
        def patched_train_step(batch):
            x_batch, y_batch = batch
            x_batch = x_batch.to(trainer.device)
            y_batch = y_batch.to(trainer.device)

            trainer.optimizer.zero_grad()
            output = trainer.model.forward(x_batch)
            # Flatten for loss
            output_flat = output.reshape(-1, output.size(-1))
            target_flat = y_batch.reshape(-1)
            loss = trainer.criterion(output_flat, target_flat)

            trainer.callbacks.on_forward_end(trainer, loss)
            loss.backward()
            trainer.callbacks.on_backward_end(trainer)
            trainer._clip_gradients()
            trainer.optimizer.step()
            trainer.callbacks.on_step_end(trainer)

            return loss.item()

        trainer._train_one_step = patched_train_step

        # Run training
        trainer.train(train_loader=loader, num_epochs=1)

        # Verify gradient monitor detected decoder.layers
        assert grad_monitor.gradient_monitor is not None, \
            "Gradient monitor not initialized"

        assert grad_monitor.block_name == 'decoder', \
            f"Expected block_name='decoder', got '{grad_monitor.block_name}'"

        assert grad_monitor.current_stats is not None, \
            "No gradient stats captured"

        print(f"  Block detected: {grad_monitor.block_name}")
        print(f"  Num layers: {len(grad_monitor.monitorable_blocks['decoder'])}")
        print(f"  Stats captured: {grad_monitor.current_stats is not None}")
        print("PASS: Gradient monitor works with decoder.layers\n")


def test_gradient_monitor_with_encoder_layers():
    """Test GradientMonitorCallback with model.encoder.layers structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Gradient Monitor with Encoder Layers")
        print("=" * 70)

        # Create a model with encoder.layers structure
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128)
                    for _ in range(2)
                ])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.encoder = MockEncoder()
                self.output = nn.Linear(64, 100)

            def forward(self, x, y=None):
                embedded = self.embedding(x)
                encoded = self.encoder(embedded)
                logits = self.output(encoded)
                return torch.log_softmax(logits, dim=-1)

        model = MockTransformer()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.NLLLoss()

        grad_monitor = GradientMonitorCallback(
            monitor_at_snapshots=False,
            monitor_interval=1,
            monitor_at_epochs=False
        )

        config = TrainerConfig(
            model_name="test_encoder_layers",
            result_dir=tmpdir,
            log_interval=100,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            callbacks=[grad_monitor]
        )

        # Create data
        x = torch.randint(0, 100, (16, 10))
        y = torch.randint(0, 100, (16, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        # Monkey-patch
        def patched_train_step(batch):
            x_batch, y_batch = batch
            x_batch = x_batch.to(trainer.device)
            y_batch = y_batch.to(trainer.device)

            trainer.optimizer.zero_grad()
            output = trainer.model.forward(x_batch)
            output_flat = output.reshape(-1, output.size(-1))
            target_flat = y_batch.reshape(-1)
            loss = trainer.criterion(output_flat, target_flat)

            trainer.callbacks.on_forward_end(trainer, loss)
            loss.backward()
            trainer.callbacks.on_backward_end(trainer)
            trainer._clip_gradients()
            trainer.optimizer.step()
            trainer.callbacks.on_step_end(trainer)

            return loss.item()

        trainer._train_one_step = patched_train_step

        # Run training
        trainer.train(train_loader=loader, num_epochs=1)

        # Verify
        assert grad_monitor.gradient_monitor is not None
        assert grad_monitor.block_name == 'encoder', \
            f"Expected block_name='encoder', got '{grad_monitor.block_name}'"

        print(f"  Block detected: {grad_monitor.block_name}")
        print(f"  Num layers: {len(grad_monitor.monitorable_blocks['encoder'])}")
        print("PASS: Gradient monitor works with encoder.layers\n")


def test_multiple_callbacks_with_real_model():
    """Test multiple callbacks working together on real model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 70)
        print("TEST: Multiple Callbacks with Real Model")
        print("=" * 70)

        # Create model with decoder.layers
        class MockDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.TransformerDecoderLayer(d_model=32, nhead=2, dim_feedforward=64)
                    for _ in range(2)
                ])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x, x)
                return x

        class MockLanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(50, 32)
                self.decoder = MockDecoder()
                self.output = nn.Linear(32, 50)

            def forward(self, x, y=None):
                embedded = self.embedding(x)
                decoded = self.decoder(embedded)
                logits = self.output(decoded)
                return torch.log_softmax(logits, dim=-1)

        model = MockLanguageModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # Very low LR
        criterion = nn.NLLLoss()

        # Multiple callbacks
        early_stop = EarlyStoppingCallback(patience=2, min_delta=0.0, restore_best=False)
        grad_monitor = GradientMonitorCallback(monitor_interval=1)

        config = TrainerConfig(
            model_name="test_multi_callbacks",
            result_dir=tmpdir,
            log_interval=100,
            snapshot_per_epoch=False,
            delete_snapshots_after_training=True
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            callbacks=[early_stop, grad_monitor]
        )

        # Create data
        x = torch.randint(0, 50, (32, 8))
        y = torch.randint(0, 50, (32, 8))
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=8)
        val_loader = DataLoader(dataset, batch_size=8)

        # Monkey-patch
        def patched_train_step(batch):
            x_batch, y_batch = batch
            x_batch = x_batch.to(trainer.device)
            y_batch = y_batch.to(trainer.device)

            trainer.optimizer.zero_grad()
            output = trainer.model.forward(x_batch)
            output_flat = output.reshape(-1, output.size(-1))
            target_flat = y_batch.reshape(-1)
            loss = trainer.criterion(output_flat, target_flat)

            trainer.callbacks.on_forward_end(trainer, loss)
            loss.backward()
            trainer.callbacks.on_backward_end(trainer)
            trainer._clip_gradients()
            trainer.optimizer.step()
            trainer.callbacks.on_step_end(trainer)

            return loss.item()

        trainer._train_one_step = patched_train_step

        # Run training (should stop early due to no improvement)
        result = trainer.train(train_loader=train_loader, val_loader=val_loader, num_epochs=10)

        # Verify both callbacks worked
        assert result['total_epochs'] < 10, \
            "Early stopping should have triggered before 10 epochs"

        assert grad_monitor.gradient_monitor is not None, \
            "Gradient monitor should be initialized"

        assert grad_monitor.current_stats is not None, \
            "Gradient stats should be captured"

        print(f"  Training stopped at epoch: {result['total_epochs']}")
        print(f"  Gradient monitor active: {grad_monitor.gradient_monitor is not None}")
        print(f"  Early stopping counter: {early_stop.counter}")
        print("PASS: Multiple callbacks work together\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INTEGRATION TESTS WITH REAL MODELS")
    print("=" * 70)

    try:
        test_gradient_monitor_with_decoder_layers()
        test_gradient_monitor_with_encoder_layers()
        test_multiple_callbacks_with_real_model()

        print("=" * 70)
        print("ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
