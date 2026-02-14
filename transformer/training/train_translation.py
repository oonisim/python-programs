"""Translation Model Training Director.

This module orchestrates the encoder-decoder Transformer training pipeline
for sequence-to-sequence translation using the Director pattern (GoF).

Architecture
------------
The training system follows clean separation of concerns:

    common.py               # Transformer components (MHA, FFN, etc.)
    encoder.py              # Encoder stack
    decoder.py              # Decoder stack
    model.py                # Encoder-Decoder Transformer
    loader_translation.py   # Translation data loading with dual tokenizers
    trainer.py              # Trainer (encoder-decoder mode)
    train_translation.py    # Training Director (this file)

Pipeline Flow
-------------
The Director orchestrates the following pipeline:

    CLI args -> Director -> Source Tokenizer + Target Tokenizer
                          -> TranslationDataLoaderFactory (loader_translation.py)
                          -> Transformer (model.py)
                          -> Trainer (trainer.py)
                          -> Training execution
                          -> Model saving

Director Steps:
    1. _build_tokenizers()   - Create source and target tokenizers
    2. _build_data_loaders() - Load parallel corpus, tokenize, create DataLoaders
    3. _build_model()        - Create Transformer (encoder-decoder)
    4. _build_trainer()      - Create optimizer, scheduler, Trainer
    5. _execute_training()   - Run training loop with checkpointing

Usage
-----
Basic training (EN -> ES):
    python train_translation.py --source_language en --target_language es

Quick test with fewer samples:
    python train_translation.py --max_samples 100 --epochs 1

Different tokenizers for source and target:
    python train_translation.py --source_tokenizer gpt4 --target_tokenizer gpt2

Resume from checkpoint:
    python train_translation.py --resume

Show this documentation:
    python train_translation.py --info

Available Datasets
------------------
    - opus_books: OPUS Books parallel corpus
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Tokenizer

from model.model import Transformer
from training.loader_translation import TranslationDataLoaderFactory, DataLoaderConfig
from training.trainer import Trainer, TrainerConfig
from training.train_lm import TiktokenAdapter, TOKENIZER_CONFIGS


# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Translation model architecture configuration."""
    d_model: int = 256
    encoder_num_heads: int = 4
    encoder_num_layers: int = 4
    encoder_d_ff: int = 512
    decoder_num_heads: int = 4
    decoder_num_layers: int = 4
    decoder_d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    log_interval: int = 100
    snapshot_interval: int = 0
    keep_last_n_snapshots: int = 3
    delete_snapshots_after_training: bool = True
    max_steps: Optional[int] = None


DATASET_CONFIGS = {
    "opus_books": "opus_books",
}


# --------------------------------------------------------------------------------
# Director
# --------------------------------------------------------------------------------
class TranslationTrainingDirector:
    """Director class orchestrating the translation training pipeline.

    Coordinates the training process without knowing internal details
    of the components. Each step in the pipeline is clearly defined.
    """

    def __init__(
            self,
            dataset_key: str,
            source_language: str,
            target_language: str,
            model_config: ModelConfig,
            training_config: TrainingConfig,
            source_tokenizer_name: str = "gpt2",
            target_tokenizer_name: str = "gpt2",
            max_samples: Optional[int] = None,
            resume: bool = False,
    ):
        """Initialize the director.

        Args:
            dataset_key: Dataset identifier (e.g., "opus_books").
            source_language: Source language code (e.g., "en").
            target_language: Target language code (e.g., "es").
            model_config: Model architecture configuration.
            training_config: Training hyperparameters.
            source_tokenizer_name: Tokenizer for source language.
            target_tokenizer_name: Tokenizer for target language.
            max_samples: Limit training samples (for quick testing).
            resume: Whether to resume from checkpoint.
        """
        self.dataset_key = dataset_key
        self.source_language = source_language
        self.target_language = target_language
        self.model_config = model_config
        self.training_config = training_config
        self.source_tokenizer_name = source_tokenizer_name
        self.target_tokenizer_name = target_tokenizer_name
        self.max_samples = max_samples
        self.resume = resume

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Components (built during pipeline execution)
        self.source_tokenizer = None
        self.target_tokenizer = None
        self.data_factory = None
        self.model = None
        self.trainer = None

    def run(self) -> dict:
        """Execute the complete training pipeline.

        Returns:
            Dictionary with training results.
        """
        if self.resume:
            self._load_run_config()
        self._print_header()
        self._build_tokenizers()
        self._build_data_loaders()
        self._build_model()
        self._build_trainer()
        if not self.resume:
            self._save_run_config()
        return self._execute_training()

    def _model_name(self) -> str:
        """Model directory name."""
        return (
            f"translation_{self.dataset_key}"
            f"_{self.source_language}_{self.target_language}"
        )

    def _run_config_path(self) -> Path:
        """Path to the saved run configuration file."""
        return Path(self._model_name()) / "snapshots" / "run_config.json"

    def _save_run_config(self) -> None:
        """Save run configuration to disk for safe resume."""
        config_path = self._run_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "dataset_key": self.dataset_key,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "source_tokenizer_name": self.source_tokenizer_name,
            "target_tokenizer_name": self.target_tokenizer_name,
            "max_samples": self.max_samples,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
        }
        config_path.write_text(json.dumps(config, indent=2))
        print(f"  Run config saved: {config_path}")

    def _load_run_config(self) -> None:
        """Load saved run configuration, overriding CLI args."""
        config_path = self._run_config_path()
        if not config_path.exists():
            print(f"WARNING: No saved run config at {config_path}, using CLI args.")
            return
        config = json.loads(config_path.read_text())
        self.source_tokenizer_name = config["source_tokenizer_name"]
        self.target_tokenizer_name = config["target_tokenizer_name"]
        self.max_samples = config["max_samples"]
        self.model_config = ModelConfig(**config["model_config"])
        self.training_config = TrainingConfig(**config["training_config"])
        print(f"Loaded run config from {config_path}, ignoring CLI args.")

    def _print_header(self) -> None:
        """Print training session header."""
        print("=" * 60)
        print("Translation Model Training")
        print("=" * 60)
        print(f"Dataset: {self.dataset_key}")
        print(f"Languages: {self.source_language} -> {self.target_language}")
        print(f"Device: {self.device}")
        print(f"Source tokenizer: {self.source_tokenizer_name}")
        print(f"Target tokenizer: {self.target_tokenizer_name}")
        print(f"Model: d={self.model_config.d_model}, "
              f"enc_h={self.model_config.encoder_num_heads}, "
              f"enc_L={self.model_config.encoder_num_layers}, "
              f"dec_h={self.model_config.decoder_num_heads}, "
              f"dec_L={self.model_config.decoder_num_layers}")
        print(f"Training: epochs={self.training_config.epochs}, "
              f"batch={self.training_config.batch_size}, "
              f"lr={self.training_config.learning_rate}")
        print()

    def _build_one_tokenizer(self, tokenizer_name: str):
        """Build a single tokenizer by name.

        Args:
            tokenizer_name: Key in TOKENIZER_CONFIGS.

        Returns:
            Tokenizer instance conforming to the Tokenizer protocol.
        """
        encoding_name = TOKENIZER_CONFIGS[tokenizer_name]
        if encoding_name is None:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        else:
            return TiktokenAdapter(encoding_name)

    def _build_tokenizers(self) -> None:
        """Step 1: Create source and target tokenizers."""
        print("Building tokenizers...")
        self.source_tokenizer = self._build_one_tokenizer(
            self.source_tokenizer_name
        )
        print(f"  Source ({self.source_tokenizer_name}): "
              f"vocab_size={self.source_tokenizer.vocab_size}")

        self.target_tokenizer = self._build_one_tokenizer(
            self.target_tokenizer_name
        )
        print(f"  Target ({self.target_tokenizer_name}): "
              f"vocab_size={self.target_tokenizer.vocab_size}")

    def _build_data_loaders(self) -> None:
        """Step 2: Create data loaders."""
        print("\nBuilding data loaders...")

        dataset_name = DATASET_CONFIGS[self.dataset_key]
        dataset_config = f"{self.source_language}-{self.target_language}"

        loader_config = DataLoaderConfig(
            max_seq_len=self.model_config.max_seq_len,
            batch_size=self.training_config.batch_size,
            max_train_samples=self.max_samples,
        )

        self.data_factory = TranslationDataLoaderFactory(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.target_tokenizer,
            source_lang=self.source_language,
            target_lang=self.target_language,
            config=loader_config,
        )

        # Trigger data loading to show stats
        _ = self.data_factory.get_train_loader()
        _ = self.data_factory.get_val_loader()

        stats = self.data_factory.get_stats()
        print(f"  Train pairs: {stats.get('train_pairs', 'N/A'):,}")
        print(f"  Val pairs: {stats.get('val_pairs', 'N/A'):,}")
        if "train_source_tokens" in stats:
            print(f"  Train source tokens: "
                  f"{stats['train_source_tokens']:,}")
            print(f"  Train target tokens: "
                  f"{stats['train_target_tokens']:,}")

    def _build_model(self) -> None:
        """Step 3: Create model."""
        print("\nBuilding model...")

        mc = self.model_config
        self.model = Transformer(
            encoder_vocabulary_size=self.source_tokenizer.vocab_size,
            encoder_max_time_steps=mc.max_seq_len,
            encoder_model_dimension=mc.d_model,
            encoder_pwff_dimension=mc.encoder_d_ff,
            encoder_dropout_ratio=mc.dropout,
            encoder_layers=mc.encoder_num_layers,
            decoder_vocabulary_size=self.target_tokenizer.vocab_size,
            decoder_max_time_steps=mc.max_seq_len,
            decoder_model_dimension=mc.d_model,
            decoder_pwff_dimension=mc.decoder_d_ff,
            decoder_dropout_ratio=mc.dropout,
            decoder_layers=mc.decoder_num_layers,
        )

        self.model.start_token = self.target_tokenizer.eos_token_id
        self.model.end_token = self.target_tokenizer.eos_token_id

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {num_params:,}")

    def _build_trainer(self) -> None:
        """Step 4: Create trainer with optimizer and scheduler."""
        print("\nBuilding trainer...")

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.training_config.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.training_config.epochs,
        )

        # Model returns log-probabilities. Do NOT use CrossEntropyLoss as
        # it expects logits and internally applies log-softmax + NLLLoss.
        criterion = nn.NLLLoss(
            ignore_index=self.target_tokenizer.pad_token_id
        )

        trainer_config = TrainerConfig(
            model_name=self._model_name(),
            gradient_clip=self.training_config.gradient_clip,
            log_interval=self.training_config.log_interval,
            snapshot_interval=self.training_config.snapshot_interval,
            snapshot_per_epoch=True,
            keep_last_n_snapshots=self.training_config.keep_last_n_snapshots,
            delete_snapshots_after_training=self.training_config.delete_snapshots_after_training,
            max_steps=self.training_config.max_steps,
        )

        self.trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            config=trainer_config,
            device=self.device,
            scheduler=scheduler,
        )

    def _execute_training(self) -> dict:
        """Step 5: Execute training loop.

        Returns:
            Dictionary with training results.
        """
        train_loader = self.data_factory.get_train_loader()
        val_loader = self.data_factory.get_val_loader()

        # Handle resume
        start_epoch = 0
        if self.resume:
            latest = self.trainer.find_latest_snapshot()
            if latest:
                print(f"\nResuming from: {latest}")
                checkpoint = self.trainer.load_snapshot(latest.name)
                start_epoch = checkpoint["epoch"] + 1
            else:
                print("\nNo checkpoint found, starting fresh.")

        print("\n" + "=" * 60)
        print("Training...")
        print("=" * 60)

        result = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.training_config.epochs,
            start_epoch=start_epoch,
        )

        self._print_summary(result)
        return result

    def _print_summary(self, result: dict) -> None:
        """Print training summary."""
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best validation loss: {result['best_val_loss']:.4f}")

        perplexity = torch.exp(torch.tensor(result["best_val_loss"])).item()
        print(f"Best perplexity: {perplexity:.2f}")
        print(f"Model saved: {result['model_path']}")


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------
HELP_TEXT = """
Translation Model Training Script
==================================

This script trains an encoder-decoder Transformer for sequence-to-sequence
translation. The encoder processes the source language and the decoder
generates the target language using cross-attention to the encoder output.

WHAT IS A TRANSLATION MODEL?
-----------------------------
A translation model converts text from one language to another. Given
an English sentence "The cat sat on the mat", it produces the Spanish
translation "El gato se sento en la alfombra".

This is the architecture from "Attention Is All You Need" (Vaswani et al., 2017).

HOW IT WORKS
------------
1. Source text is tokenized using the source tokenizer
2. Target text is tokenized using the target tokenizer
3. The encoder processes source tokens through self-attention layers
4. The decoder processes target tokens with causal self-attention
   and cross-attention to the encoder output
5. Training uses teacher forcing: the decoder receives the ground-truth
   target shifted right, and predicts the next token at each position

QUICK START
-----------
# Train with default settings (EN -> ES, opus_books)
python train_translation.py

# Quick test with fewer samples
python train_translation.py --max_samples 100 --epochs 1

# Different language pair
python train_translation.py --source_language fr --target_language en

# Different tokenizers for each side
python train_translation.py --source_tokenizer gpt4 --target_tokenizer gpt2

# Resume interrupted training
python train_translation.py --resume

OUTPUT
------
After training, the model is saved to:
    translation_<dataset>_<src>_<tgt>/models/model_<timestamp>.pt
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an encoder-decoder Transformer for translation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use --info for detailed documentation, --help for argument help.",
    )

    # --------------------------------------------------------------------------------
    # Data Arguments
    # --------------------------------------------------------------------------------
    data_group = parser.add_argument_group(
        "Data",
        "Dataset, language, and tokenizer options.",
    )
    data_group.add_argument(
        "--dataset", type=str, default="opus_books",
        choices=list(DATASET_CONFIGS.keys()),
        help=(
            "Dataset to train on. "
            "'opus_books' uses OPUS Books parallel corpus. "
            "Default: opus_books"
        ),
    )
    data_group.add_argument(
        "--source_language", type=str.lower, default="en",
        help=(
            "Source language code (case-insensitive). "
            "Examples: en, fr, de, es. Default: en"
        ),
    )
    data_group.add_argument(
        "--target_language", type=str.lower, default="es",
        help=(
            "Target language code (case-insensitive). "
            "Examples: es, fr, de, en. Default: es"
        ),
    )
    data_group.add_argument(
        "--source_tokenizer", type=str, default="gpt2",
        choices=list(TOKENIZER_CONFIGS.keys()),
        help=(
            "Tokenizer for source language. "
            "'gpt2' uses GPT-2 BPE (50257 tokens). "
            "'gpt4' uses tiktoken cl100k_base (100256 tokens). "
            "'gpt4o' uses tiktoken o200k_base (200019 tokens). "
            "Default: gpt2"
        ),
    )
    data_group.add_argument(
        "--target_tokenizer", type=str, default="gpt2",
        choices=list(TOKENIZER_CONFIGS.keys()),
        help=(
            "Tokenizer for target language. "
            "'gpt2' uses GPT-2 BPE (50257 tokens). "
            "'gpt4' uses tiktoken cl100k_base (100256 tokens). "
            "'gpt4o' uses tiktoken o200k_base (200019 tokens). "
            "Default: gpt2"
        ),
    )
    data_group.add_argument(
        "--max_samples", type=int, default=None,
        metavar="N",
        help=(
            "Limit the number of training pairs. Useful for quick testing "
            "or debugging. Default: None (use all pairs)"
        ),
    )

    # --------------------------------------------------------------------------------
    # Training Arguments
    # --------------------------------------------------------------------------------
    train_group = parser.add_argument_group(
        "Training",
        "Training hyperparameters controlling the optimization process.",
    )
    train_group.add_argument(
        "--epochs", type=int, default=20,
        metavar="N",
        help="Number of training epochs. Default: 20",
    )
    train_group.add_argument(
        "--batch_size", type=int, default=32,
        metavar="N",
        help=(
            "Number of sequence pairs per gradient update. Reduce if "
            "you encounter OOM errors. Default: 32"
        ),
    )
    train_group.add_argument(
        "--lr", type=float, default=3e-4,
        metavar="RATE",
        help="Learning rate for AdamW optimizer. Default: 3e-4",
    )
    train_group.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume training from the latest checkpoint. The script "
            "automatically finds the most recent snapshot."
        ),
    )
    train_group.add_argument(
        "--max_steps", type=int, default=None,
        metavar="N",
        help=(
            "Maximum number of training steps. Training stops after N steps "
            "regardless of epochs. Default: None (no limit)"
        ),
    )
    train_group.add_argument(
        "--snapshot_interval", type=int, default=0,
        metavar="N",
        help=(
            "Save a snapshot every N training steps. "
            "0 disables step-based snapshots. Default: 0"
        ),
    )
    train_group.add_argument(
        "--keep_last_n_snapshots", type=int, default=3,
        metavar="N",
        help=(
            "Number of recent snapshots to keep on disk. "
            "0 keeps all snapshots. Default: 3"
        ),
    )
    train_group.add_argument(
        "--delete_snapshots_after_training", action="store_true",
        default=True,
        help="Delete all snapshots after training completes. Enabled by default.",
    )
    train_group.add_argument(
        "--no_delete_snapshots_after_training", action="store_false",
        dest="delete_snapshots_after_training",
        help="Keep snapshots after training completes.",
    )

    # --------------------------------------------------------------------------------
    # Model Architecture Arguments
    # --------------------------------------------------------------------------------
    model_group = parser.add_argument_group(
        "Model Architecture",
        "Transformer architecture hyperparameters.",
    )
    model_group.add_argument(
        "--d_model", type=int, default=256,
        metavar="DIM",
        help="Model embedding dimension (shared by encoder and decoder). Default: 256",
    )
    model_group.add_argument(
        "--encoder_num_heads", type=int, default=4,
        metavar="N",
        help="Number of attention heads in encoder. Default: 4",
    )
    model_group.add_argument(
        "--encoder_num_layers", type=int, default=4,
        metavar="N",
        help="Number of encoder layers. Default: 4",
    )
    model_group.add_argument(
        "--encoder_d_ff", type=int, default=512,
        metavar="DIM",
        help="Feed-forward hidden dimension in encoder. Default: 512",
    )
    model_group.add_argument(
        "--decoder_num_heads", type=int, default=4,
        metavar="N",
        help="Number of attention heads in decoder. Default: 4",
    )
    model_group.add_argument(
        "--decoder_num_layers", type=int, default=4,
        metavar="N",
        help="Number of decoder layers. Default: 4",
    )
    model_group.add_argument(
        "--decoder_d_ff", type=int, default=512,
        metavar="DIM",
        help="Feed-forward hidden dimension in decoder. Default: 512",
    )
    model_group.add_argument(
        "--max_seq_len", type=int, default=128,
        metavar="LEN",
        help="Maximum sequence length for both source and target. Default: 128",
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.1,
        metavar="RATE",
        help="Dropout probability for regularization. Default: 0.1",
    )

    # --------------------------------------------------------------------------------
    # Information Arguments
    # --------------------------------------------------------------------------------
    info_group = parser.add_argument_group(
        "Information",
        "Help and documentation options.",
    )
    info_group.add_argument(
        "--info", action="store_true",
        help="Show detailed documentation about this script.",
    )
    info_group.add_argument(
        "--explain", action="store_true",
        help="Show explanation and usage examples.",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.info:
        print(__doc__)
        sys.exit(0)

    if args.explain:
        print(HELP_TEXT)
        sys.exit(0)

    model_config = ModelConfig(
        d_model=args.d_model,
        encoder_num_heads=args.encoder_num_heads,
        encoder_num_layers=args.encoder_num_layers,
        encoder_d_ff=args.encoder_d_ff,
        decoder_num_heads=args.decoder_num_heads,
        decoder_num_layers=args.decoder_num_layers,
        decoder_d_ff=args.decoder_d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        snapshot_interval=args.snapshot_interval,
        keep_last_n_snapshots=args.keep_last_n_snapshots,
        delete_snapshots_after_training=args.delete_snapshots_after_training,
        max_steps=args.max_steps,
    )

    director = TranslationTrainingDirector(
        dataset_key=args.dataset,
        source_language=args.source_language,
        target_language=args.target_language,
        model_config=model_config,
        training_config=training_config,
        source_tokenizer_name=args.source_tokenizer,
        target_tokenizer_name=args.target_tokenizer,
        max_samples=args.max_samples,
        resume=args.resume,
    )

    director.run()


if __name__ == "__main__":
    main()
