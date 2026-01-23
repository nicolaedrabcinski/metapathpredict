"""
Dagster assets for MLOps pipeline orchestration.

Defines software-defined assets for:
- Data preparation
- Model training
- Inference
- Evaluation
"""

from __future__ import annotations

from typing import Any

try:
    from dagster import (
        AssetExecutionContext,
        MetadataValue,
        Output,
        asset,
        define_asset_job,
        get_dagster_logger,
    )
    DAGSTER_AVAILABLE = True
except ImportError:
    DAGSTER_AVAILABLE = False


def _check_dagster():
    """Check if Dagster is available."""
    if not DAGSTER_AVAILABLE:
        raise ImportError("Dagster not installed. Run: pip install dagster")


# Only define assets if Dagster is available
if DAGSTER_AVAILABLE:
    
    @asset(
        description="Prepare training dataset from FASTA files",
        group_name="data",
        compute_kind="python",
    )
    def prepare_dataset_asset(context: AssetExecutionContext) -> Output:
        """
        Prepare dataset from raw FASTA files.
        
        Reads FASTA files, encodes sequences, and saves to HDF5/Parquet.
        """
        logger = get_dagster_logger()
        
        from metapathpredict.config import Settings
        from metapathpredict.data import SequencePreprocessor
        
        import h5py
        import numpy as np
        
        # Load config
        settings = Settings()
        
        # Initialize preprocessor
        preprocessor = SequencePreprocessor(
            sequence_length=settings.data.sequence_length,
        )
        
        # Process each class
        class_files = {
            "bacteria": settings.paths.data_dir / "input" / "bacteria.fasta",
            "eukaryotic": settings.paths.data_dir / "input" / "eucaryotic.fasta",
            "virus": settings.paths.data_dir / "input" / "viruses.fasta",
        }
        
        class_mapping = {"bacteria": 0, "eukaryotic": 1, "virus": 2}
        
        all_data = []
        all_labels = []
        
        for class_name, fasta_path in class_files.items():
            if not fasta_path.exists():
                logger.warning(f"File not found: {fasta_path}")
                continue
            
            # Read FASTA
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
            
            logger.info(f"Processing {class_name}: {len(sequences)} sequences")
            
            # Encode
            for seq in sequences:
                try:
                    encoded = preprocessor.encode_sequence(seq)
                    all_data.append(encoded)
                    all_labels.append(class_mapping[class_name])
                except Exception as e:
                    logger.warning(f"Failed to encode: {e}")
        
        # Save to HDF5
        output_dir = settings.paths.data_dir / "datasets" / "unified"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_array = np.stack(all_data, axis=0)
        labels_array = np.array(all_labels, dtype=np.int64)
        
        output_path = output_dir / f"encoded_train_{settings.data.sequence_length}.hdf5"
        
        with h5py.File(output_path, "w") as f:
            f.create_dataset("data", data=data_array, compression="gzip")
            f.create_dataset("labels", data=labels_array, compression="gzip")
        
        logger.info(f"Saved dataset: {output_path}")
        
        return Output(
            value=str(output_path),
            metadata={
                "num_samples": MetadataValue.int(len(all_data)),
                "sequence_length": MetadataValue.int(settings.data.sequence_length),
                "class_distribution": MetadataValue.json({
                    "bacteria": int((labels_array == 0).sum()),
                    "eukaryotic": int((labels_array == 1).sum()),
                    "virus": int((labels_array == 2).sum()),
                }),
            },
        )
    
    
    @asset(
        description="Train sequence classification model",
        deps=[prepare_dataset_asset],
        group_name="training",
        compute_kind="pytorch",
    )
    def train_model_asset(context: AssetExecutionContext) -> Output:
        """
        Train model on prepared dataset.
        """
        logger = get_dagster_logger()
        
        import time
        import uuid
        
        import torch
        
        from metapathpredict.config import Settings
        from metapathpredict.data import SequenceDataModule
        from metapathpredict.models import UnifiedClassifier
        from metapathpredict.training import (
            EarlyStopping,
            ModelCheckpoint,
            Trainer,
            get_scheduler,
        )
        
        settings = Settings()
        
        # Create data module
        data_module = SequenceDataModule.from_config(settings)
        data_module.setup()
        
        # Create model
        model = UnifiedClassifier(
            in_channels=4,
            num_classes=3,
            sequence_length=settings.data.sequence_length,
        )
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.training.learning_rate,
            weight_decay=settings.training.weight_decay,
        )
        
        scheduler = get_scheduler(
            name="warmup_cosine",
            optimizer=optimizer,
            total_epochs=settings.training.epochs,
            warmup_epochs=5,
        )
        
        # Callbacks
        output_dir = settings.paths.weights_dir / "unified"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10),
            ModelCheckpoint(save_dir=output_dir, monitor="val_loss"),
        ]
        
        # Train
        trainer = Trainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            optimizer=optimizer,
            scheduler=scheduler,
            config=settings.training,
            callbacks=callbacks,
        )
        
        start_time = time.time()
        history = trainer.fit()
        duration = time.time() - start_time
        
        # Save final model
        model_path = output_dir / "final_model.pt"
        trainer.save_checkpoint(model_path)
        
        # Log to DuckDB if available
        try:
            from metapathpredict.pipeline.duckdb_connector import DuckDBConnector
            
            db = DuckDBConnector(str(settings.paths.data_dir / "metadata.duckdb"))
            db.create_sequences_table()
            db.log_training_run(
                run_id=str(uuid.uuid4()),
                model_type="UnifiedClassifier",
                config=settings.training.model_dump(),
                train_loss=history["train_loss"][-1],
                val_loss=history["val_loss"][-1],
                val_accuracy=history["val_accuracy"][-1],
                epochs=len(history["train_loss"]),
                duration=duration,
            )
            db.close()
        except Exception as e:
            logger.warning(f"Could not log to DuckDB: {e}")
        
        return Output(
            value=str(model_path),
            metadata={
                "final_train_loss": MetadataValue.float(history["train_loss"][-1]),
                "final_val_loss": MetadataValue.float(history["val_loss"][-1]),
                "final_val_accuracy": MetadataValue.float(history["val_accuracy"][-1]),
                "epochs_trained": MetadataValue.int(len(history["train_loss"])),
                "duration_seconds": MetadataValue.float(duration),
            },
        )
    
    
    @asset(
        description="Run inference on test data",
        deps=[train_model_asset],
        group_name="inference",
        compute_kind="pytorch",
    )
    def predict_asset(context: AssetExecutionContext) -> Output:
        """
        Run inference on test dataset.
        """
        logger = get_dagster_logger()
        
        import numpy as np
        
        from metapathpredict.config import Settings
        from metapathpredict.data import HDF5SequenceDataset
        from metapathpredict.inference import Predictor
        from metapathpredict.models import UnifiedClassifier
        from torch.utils.data import DataLoader
        
        settings = Settings()
        
        # Load model
        model_path = settings.paths.weights_dir / "unified" / "best_model.pt"
        
        predictor = Predictor.from_checkpoint(
            checkpoint_path=model_path,
            model_class=UnifiedClassifier,
            model_kwargs={
                "in_channels": 4,
                "num_classes": 3,
                "sequence_length": settings.data.sequence_length,
            },
        )
        
        # Load test data
        test_path = settings.paths.data_dir / "datasets" / "unified" / f"encoded_test_{settings.data.sequence_length}.hdf5"
        
        test_dataset = HDF5SequenceDataset(str(test_path))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Predict
        results = predictor.predict_dataloader(test_loader, use_tta=True)
        
        accuracy = results.get("accuracy", 0)
        
        # Save predictions
        output_path = settings.paths.data_dir / "output" / "predictions" / "test_predictions.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_path,
            predictions=results["predicted_class"],
            probabilities=results["probabilities"],
            targets=results.get("targets"),
        )
        
        logger.info(f"Predictions saved to {output_path}")
        
        return Output(
            value=str(output_path),
            metadata={
                "num_predictions": MetadataValue.int(len(results["predicted_class"])),
                "accuracy": MetadataValue.float(accuracy),
                "avg_confidence": MetadataValue.float(float(np.mean(results["confidence"]))),
            },
        )
    
    
    @asset(
        description="Evaluate model performance",
        deps=[predict_asset],
        group_name="evaluation",
        compute_kind="python",
    )
    def evaluate_asset(context: AssetExecutionContext) -> Output:
        """
        Evaluate model and generate metrics.
        """
        logger = get_dagster_logger()
        
        import json
        
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix
        
        from metapathpredict.config import Settings
        
        settings = Settings()
        
        # Load predictions
        pred_path = settings.paths.data_dir / "output" / "predictions" / "test_predictions.npz"
        
        data = np.load(pred_path)
        predictions = data["predictions"]
        targets = data["targets"]
        probabilities = data["probabilities"]
        
        class_names = ["bacteria", "eukaryotic", "virus"]
        
        # Calculate metrics
        accuracy = (predictions == targets).mean()
        
        report = classification_report(
            targets,
            predictions,
            target_names=class_names,
            output_dict=True,
        )
        
        cm = confusion_matrix(targets, predictions)
        
        # Save evaluation report
        output_path = settings.paths.data_dir / "output" / "evaluation_report.json"
        
        eval_results = {
            "accuracy": float(accuracy),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        
        with open(output_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        # Log to DataHub if available
        try:
            from datahub.emitter.mce_builder import make_dataset_urn
            from datahub.emitter.rest_emitter import DatahubRestEmitter
            
            emitter = DatahubRestEmitter("http://localhost:8080")
            # Could emit metadata here
        except Exception:
            pass
        
        return Output(
            value=str(output_path),
            metadata={
                "accuracy": MetadataValue.float(accuracy),
                "f1_bacteria": MetadataValue.float(report["bacteria"]["f1-score"]),
                "f1_eukaryotic": MetadataValue.float(report["eukaryotic"]["f1-score"]),
                "f1_virus": MetadataValue.float(report["virus"]["f1-score"]),
                "confusion_matrix": MetadataValue.json(cm.tolist()),
            },
        )
    
    
    # Define job
    training_pipeline_job = define_asset_job(
        name="training_pipeline",
        selection=[
            prepare_dataset_asset,
            train_model_asset,
            predict_asset,
            evaluate_asset,
        ],
        description="Full ML training pipeline",
    )

else:
    # Dummy functions when Dagster not available
    def prepare_dataset_asset(*args, **kwargs):
        raise ImportError("Dagster not installed")
    
    def train_model_asset(*args, **kwargs):
        raise ImportError("Dagster not installed")
    
    def predict_asset(*args, **kwargs):
        raise ImportError("Dagster not installed")
    
    def evaluate_asset(*args, **kwargs):
        raise ImportError("Dagster not installed")
