"""
Command-line interface for metapathpredict.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_device(args: argparse.Namespace) -> torch.device:
    """Set up compute device."""
    if args.device:
        return torch.device(args.device)
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_command(args: argparse.Namespace) -> int:
    """Train a model."""
    from metapathpredict.config import Settings
    from metapathpredict.data import SequenceDataModule
    from metapathpredict.models import UnifiedClassifier, create_cnn_model
    from metapathpredict.training import (
        EarlyStopping,
        MetricsLogger,
        ModelCheckpoint,
        ProgressCallback,
        Trainer,
        get_scheduler,
    )
    
    # Load config
    if args.config:
        settings = Settings.from_yaml(args.config)
    else:
        settings = Settings()
    
    # Override with command-line arguments
    if args.epochs:
        settings.training.num_epochs = args.epochs
    if args.batch_size:
        settings.training.batch_size = args.batch_size
    if args.learning_rate:
        settings.training.learning_rate = args.learning_rate
    
    device = setup_device(args)
    logger.info(f"Training on {device}")
    
    # Create data module
    data_module = SequenceDataModule.from_config(settings)
    
    # Create model
    if args.model == "unified":
        model = UnifiedClassifier(
            seq_length=settings.data.default_fragment_size,
            num_classes=settings.model.num_classes,
        )
    else:
        model = create_cnn_model(
            model_type=args.model,
            in_channels=4,
            num_classes=settings.model.num_classes,
        )
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )
    
    # Create scheduler
    scheduler = get_scheduler(
        name=settings.training.scheduler or "warmup_cosine",
        optimizer=optimizer,
        total_epochs=settings.training.num_epochs,
        warmup_epochs=settings.training.warmup_epochs,
    )
    
    # Create callbacks
    output_dir = Path(args.output or settings.paths.weights_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ProgressCallback(show_metrics=["accuracy", "f1_macro"]),
        MetricsLogger(log_dir=output_dir),
        ModelCheckpoint(
            save_dir=output_dir,
            monitor="val_loss",
            save_top_k=3,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=settings.training.patience,
        ),
    ]
    
    # Get class weights if enabled
    class_weights = None
    if settings.training.use_class_weights:
        class_weights = data_module.get_class_weights(device)
        logger.info(f"Using computed class weights: {class_weights.tolist()}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=settings.training,
        callbacks=callbacks,
        device=device,
        class_weights=class_weights,
    )
    
    # Train
    history = trainer.fit(num_epochs=settings.training.num_epochs)
    
    # Save final model
    trainer.save_checkpoint(output_dir / "final_model.pt")
    
    logger.info(f"Training complete. Models saved to {output_dir}")
    
    return 0


def predict_command(args: argparse.Namespace) -> int:
    """Run inference on input data."""
    from metapathpredict.config import Settings
    from metapathpredict.data import SequencePreprocessor
    from metapathpredict.inference import Predictor
    from metapathpredict.models import UnifiedClassifier
    
    # Load config if provided
    if args.config:
        settings = Settings.from_yaml(args.config)
    else:
        settings = Settings()
    
    device = setup_device(args)
    logger.info(f"Inference on {device}")
    
    # Load model
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        logger.error(f"Model checkpoint not found: {checkpoint_path}")
        return 1
    
    # Create predictor
    predictor = Predictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_class=UnifiedClassifier,
        model_kwargs={
            "in_channels": 4,
            "num_classes": 3,
            "sequence_length": settings.data.default_fragment_size,
        },
        device=device,
    )
    
    # Process input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Create preprocessor
    preprocessor = SequencePreprocessor(
        sequence_length=settings.data.default_fragment_size,
    )
    
    # Read sequences
    if input_path.suffix in (".fasta", ".fa", ".fna"):
        sequences = []
        seq_ids = []
        
        current_id = None
        current_seq = []
        
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id is not None:
                        sequences.append("".join(current_seq))
                        seq_ids.append(current_id)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            
            if current_id is not None:
                sequences.append("".join(current_seq))
                seq_ids.append(current_id)
        
        logger.info(f"Loaded {len(sequences)} sequences from {input_path}")
    else:
        logger.error(f"Unsupported input format: {input_path.suffix}")
        return 1
    
    # Make predictions
    results = predictor.predict_sequences(
        sequences=sequences,
        preprocessor=preprocessor,
        batch_size=args.batch_size or 32,
        use_tta=args.tta,
    )
    
    # Output results
    output_path = Path(args.output) if args.output else input_path.with_suffix(".predictions.tsv")
    
    with open(output_path, "w") as f:
        # Header
        f.write("sequence_id\tpredicted_class\tconfidence\t")
        f.write("\t".join([f"prob_{c}" for c in predictor.class_names]))
        f.write("\n")
        
        # Data
        for i, seq_id in enumerate(seq_ids):
            pred_label = results["predicted_label"][i]
            confidence = results["confidence"][i]
            probs = results["probabilities"][i]
            
            f.write(f"{seq_id}\t{pred_label}\t{confidence:.4f}\t")
            f.write("\t".join([f"{p:.4f}" for p in probs]))
            f.write("\n")
    
    logger.info(f"Predictions saved to {output_path}")
    
    # Print summary
    print("\nPrediction Summary:")
    print("-" * 40)
    for class_name in predictor.class_names:
        count = sum(1 for l in results["predicted_label"] if l == class_name)
        print(f"  {class_name}: {count} ({100 * count / len(sequences):.1f}%)")
    print(f"  Average confidence: {np.mean(results['confidence']):.4f}")
    
    return 0


def prepare_command(args: argparse.Namespace) -> int:
    """Prepare dataset from FASTA files."""
    from metapathpredict.config import Settings
    from metapathpredict.data import SequencePreprocessor
    
    import h5py
    
    # Load config
    if args.config:
        settings = Settings.from_yaml(args.config)
    else:
        settings = Settings()
    
    # Override with args
    sequence_length = args.length or settings.data.default_fragment_size
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create preprocessor
    preprocessor = SequencePreprocessor(sequence_length=sequence_length)
    
    # Process each input file
    class_mapping = {
        "bacteria": 0,
        "eukaryotic": 1,
        "virus": 2,
        "viruses": 2,
    }
    
    all_data = []
    all_labels = []
    
    for fasta_path in args.inputs:
        fasta_path = Path(fasta_path)
        
        if not fasta_path.exists():
            logger.error(f"File not found: {fasta_path}")
            continue
        
        # Determine class from filename
        class_name = None
        for key in class_mapping:
            if key in fasta_path.stem.lower():
                class_name = key
                break
        
        if class_name is None:
            logger.warning(f"Could not determine class for {fasta_path}")
            continue
        
        class_label = class_mapping[class_name]
        logger.info(f"Processing {fasta_path} as class {class_name} (label {class_label})")
        
        # Read and encode sequences
        sequences = []
        current_seq = []
        
        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                    current_seq = []
                else:
                    current_seq.append(line)
            
            if current_seq:
                sequences.append("".join(current_seq))
        
        logger.info(f"  Found {len(sequences)} sequences")
        
        # Encode sequences
        for seq in sequences:
            try:
                encoded = preprocessor.encode_sequence(seq)
                all_data.append(encoded)
                all_labels.append(class_label)
            except Exception as e:
                logger.warning(f"  Failed to encode sequence: {e}")
    
    if not all_data:
        logger.error("No sequences processed")
        return 1
    
    # Convert to arrays
    data_array = np.stack(all_data, axis=0)
    labels_array = np.array(all_labels, dtype=np.int64)
    
    logger.info(f"Total: {len(data_array)} sequences")
    logger.info(f"  Shape: {data_array.shape}")
    
    # Class distribution
    for class_name, label in class_mapping.items():
        count = (labels_array == label).sum()
        if count > 0:
            logger.info(f"  {class_name}: {count}")
    
    # Split into train/val/test
    from sklearn.model_selection import train_test_split
    
    # First split: 80% train, 20% temp
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data_array, labels_array,
        test_size=0.2,
        random_state=42,
        stratify=labels_array,
    )
    
    # Second split: 50% val, 50% test from temp
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels,
    )
    
    # Save to HDF5
    train_path = output_dir / f"encoded_train_{sequence_length}.hdf5"
    val_path = output_dir / f"encoded_val_{sequence_length}.hdf5"
    test_path = output_dir / f"encoded_test_{sequence_length}.hdf5"
    
    for path, data, labels, name in [
        (train_path, train_data, train_labels, "train"),
        (val_path, val_data, val_labels, "val"),
        (test_path, test_data, test_labels, "test"),
    ]:
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
        
        logger.info(f"Saved {name}: {len(data)} samples to {path}")
    
    # Save metadata
    metadata = {
        "sequence_length": sequence_length,
        "num_classes": 3,
        "class_names": ["bacteria", "eukaryotic", "virus"],
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset prepared in {output_dir}")
    
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    """Evaluate model on test data."""
    from metapathpredict.config import Settings
    from metapathpredict.data import HDF5SequenceDataset
    from metapathpredict.inference import Predictor
    from metapathpredict.models import UnifiedClassifier
    
    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader
    
    # Load config
    if args.config:
        settings = Settings.from_yaml(args.config)
    else:
        settings = Settings()
    
    device = setup_device(args)
    
    # Load model
    predictor = Predictor.from_checkpoint(
        checkpoint_path=args.model,
        model_class=UnifiedClassifier,
        model_kwargs={
            "in_channels": 4,
            "num_classes": 3,
            "sequence_length": settings.data.default_fragment_size,
        },
        device=device,
    )
    
    # Load test data
    test_dataset = HDF5SequenceDataset(args.data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size or 64,
        shuffle=False,
        num_workers=4,
    )
    
    # Predict
    results = predictor.predict_dataloader(
        test_loader,
        use_tta=args.tta,
        show_progress=True,
    )
    
    # Metrics
    predictions = results["predicted_class"]
    targets = results["targets"]
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        targets,
        predictions,
        target_names=predictor.class_names,
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(targets, predictions)
    print(cm)
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        
        eval_results = {
            "accuracy": float(results["accuracy"]),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                targets, predictions,
                target_names=predictor.class_names,
                output_dict=True,
            ),
        }
        
        with open(output_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="metapathpredict",
        description="MetaPathPredict: DNA sequence classification",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 2.0.0",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", "-c", help="Path to config file")
    train_parser.add_argument("--model", "-m", default="unified", 
                             choices=["unified", "simple", "multiscale", "residual"],
                             help="Model architecture")
    train_parser.add_argument("--epochs", "-e", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", "-lr", type=float, help="Learning rate")
    train_parser.add_argument("--output", "-o", help="Output directory")
    train_parser.add_argument("--device", help="Device (cuda/cpu)")
    train_parser.set_defaults(func=train_command)
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    predict_parser.add_argument("--input", "-i", required=True, help="Input FASTA file")
    predict_parser.add_argument("--output", "-o", help="Output file")
    predict_parser.add_argument("--config", "-c", help="Path to config file")
    predict_parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
    predict_parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    predict_parser.add_argument("--device", help="Device (cuda/cpu)")
    predict_parser.set_defaults(func=predict_command)
    
    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prepare_parser.add_argument("inputs", nargs="+", help="Input FASTA files")
    prepare_parser.add_argument("--output", "-o", required=True, help="Output directory")
    prepare_parser.add_argument("--config", "-c", help="Path to config file")
    prepare_parser.add_argument("--length", "-l", type=int, help="Sequence length")
    prepare_parser.set_defaults(func=prepare_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--data", "-d", required=True, help="Test data HDF5 file")
    eval_parser.add_argument("--output", "-o", help="Output file for results")
    eval_parser.add_argument("--config", "-c", help="Path to config file")
    eval_parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
    eval_parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    eval_parser.add_argument("--device", help="Device (cuda/cpu)")
    eval_parser.set_defaults(func=evaluate_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
