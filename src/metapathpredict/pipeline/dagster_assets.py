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
            sequence_length=settings.data.default_fragment_size,
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
        
        output_path = output_dir / f"encoded_train_{settings.data.default_fragment_size}.hdf5"
        
        with h5py.File(output_path, "w") as f:
            f.create_dataset("data", data=data_array, compression="gzip")
            f.create_dataset("labels", data=labels_array, compression="gzip")
        
        logger.info(f"Saved dataset: {output_path}")
        
        return Output(
            value=str(output_path),
            metadata={
                "num_samples": MetadataValue.int(len(all_data)),
                "sequence_length": MetadataValue.int(settings.data.default_fragment_size),
                "class_distribution": MetadataValue.json({
                    "bacteria": int((labels_array == 0).sum()),
                    "eukaryotic": int((labels_array == 1).sum()),
                    "virus": int((labels_array == 2).sum()),
                }),
            },
        )
    
    
    @asset(
        description="Train contrastive + RL pipeline",
        deps=[prepare_dataset_asset],
        group_name="training",
        compute_kind="pytorch",
    )
    def train_model_asset(context: AssetExecutionContext) -> Output:
        """
        Train full pipeline: contrastive pretraining -> RL fine-tuning.
        """
        logger = get_dagster_logger()

        import time
        import uuid

        import torch

        from metapathpredict.config import Settings
        from metapathpredict.data import SequenceDataModule
        from metapathpredict.models import (
            ActorCriticAgent,
            ContrastiveAugmentation,
            ContrastiveEncoder,
            ContrastiveTrainer,
            DQNAgent,
            PolicyGradientAgent,
            RLTrainer,
            SequenceEnvironment,
        )

        settings = Settings()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_module = SequenceDataModule.from_config(settings)
        data_module.setup()

        output_dir = settings.paths.weights_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # ── Phase 1: Contrastive Pretraining ──
        cfg_c = settings.contrastive
        encoder = ContrastiveEncoder(
            in_channels=4,
            backbone=cfg_c.backbone,
            projection_dim=cfg_c.projection_dim,
            hidden_dim=cfg_c.hidden_dim,
            base_channels=cfg_c.base_channels,
        )
        logger.info(f"ContrastiveEncoder: {sum(p.numel() for p in encoder.parameters()):,} params")

        augmentation = ContrastiveAugmentation(
            mutation_rate=cfg_c.mutation_rate,
            mask_rate=cfg_c.mask_rate,
        )
        optimizer_c = torch.optim.AdamW(
            encoder.parameters(), lr=cfg_c.learning_rate, weight_decay=cfg_c.weight_decay,
        )
        trainer_c = ContrastiveTrainer(
            encoder=encoder,
            optimizer=optimizer_c,
            augmentation=augmentation,
            temperature=cfg_c.temperature,
            use_supervised=(cfg_c.loss_type == "supcon"),
            device=device,
        )

        train_loader = data_module.train_dataloader()
        best_loss = float("inf")
        for epoch in range(cfg_c.num_epochs):
            loss = trainer_c.train_epoch(train_loader)
            logger.info(f"Contrastive Epoch {epoch+1}/{cfg_c.num_epochs} - Loss: {loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                torch.save({
                    "encoder_state_dict": encoder.state_dict(),
                    "config": cfg_c.model_dump(),
                }, output_dir / "contrastive_best.pt")

        contrastive_ckpt = output_dir / "contrastive_best.pt"

        # ── Phase 2: RL Fine-tuning ──
        cfg_r = settings.rl
        all_sequences, all_labels = [], []
        for batch in data_module.train_dataloader():
            all_sequences.append(batch[0])
            all_labels.append(batch[1])
        all_sequences = torch.cat(all_sequences, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        agent_map = {
            "dqn": DQNAgent,
            "policy_gradient": PolicyGradientAgent,
            "actor_critic": ActorCriticAgent,
        }
        AgentClass = agent_map[cfg_r.algorithm]
        agent = AgentClass(
            in_channels=4, num_actions=3,
            backbone=cfg_r.backbone, hidden_dim=cfg_r.hidden_dim,
        )

        # Transfer encoder weights
        if contrastive_ckpt.exists():
            ckpt = torch.load(contrastive_ckpt, map_location=device, weights_only=False)
            sd = ckpt["encoder_state_dict"]
            transfer = {k: v for k, v in sd.items() if k.startswith("encoder.")}
            agent.load_state_dict(transfer, strict=False)
            logger.info(f"Transferred {len(transfer)} tensors from contrastive encoder")

        env = SequenceEnvironment(
            sequences=all_sequences, labels=all_labels,
            reward_correct=cfg_r.reward_correct,
            reward_incorrect=cfg_r.reward_incorrect,
            reward_uncertain=cfg_r.reward_uncertain,
        )
        optimizer_r = torch.optim.AdamW(
            agent.parameters(), lr=cfg_r.learning_rate, weight_decay=cfg_r.weight_decay,
        )
        rl_trainer = RLTrainer(
            agent=agent, environment=env, optimizer=optimizer_r,
            device=device, gamma=cfg_r.gamma, algorithm=cfg_r.algorithm,
        )

        best_accuracy = 0.0
        for epoch in range(cfg_r.num_epochs):
            if cfg_r.algorithm == "dqn":
                frac = min(epoch / max(cfg_r.epsilon_decay_epochs, 1), 1.0)
                epsilon = cfg_r.epsilon_start + frac * (cfg_r.epsilon_end - cfg_r.epsilon_start)
            else:
                epsilon = 0.0
            metrics = rl_trainer.train_epoch(num_episodes=cfg_r.episodes_per_epoch, epsilon=epsilon)
            logger.info(
                f"RL Epoch {epoch+1}/{cfg_r.num_epochs} - "
                f"Reward: {metrics['avg_reward']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
            )
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                torch.save({
                    "agent_state_dict": agent.state_dict(),
                    "algorithm": cfg_r.algorithm,
                    "config": cfg_r.model_dump(),
                    "metrics": metrics,
                }, output_dir / "rl_best.pt")

        duration = time.time() - start_time

        # Log to DuckDB if available
        try:
            from metapathpredict.pipeline.duckdb_connector import DuckDBConnector

            db = DuckDBConnector(str(settings.paths.data_dir / "metadata.duckdb"))
            db.create_sequences_table()
            db.log_training_run(
                run_id=str(uuid.uuid4()),
                model_type=f"Contrastive+{AgentClass.__name__}",
                config={**cfg_c.model_dump(), **cfg_r.model_dump()},
                train_loss=best_loss,
                val_loss=0.0,
                val_accuracy=best_accuracy,
                epochs=cfg_c.num_epochs + cfg_r.num_epochs,
                duration=duration,
            )
            db.close()
        except Exception as e:
            logger.warning(f"Could not log to DuckDB: {e}")

        return Output(
            value=str(output_dir / "rl_best.pt"),
            metadata={
                "contrastive_best_loss": MetadataValue.float(best_loss),
                "rl_best_accuracy": MetadataValue.float(best_accuracy),
                "total_epochs": MetadataValue.int(cfg_c.num_epochs + cfg_r.num_epochs),
                "duration_seconds": MetadataValue.float(duration),
            },
        )
    
    
    @asset(
        description="Run inference on test data using RL agent",
        deps=[train_model_asset],
        group_name="inference",
        compute_kind="pytorch",
    )
    def predict_asset(context: AssetExecutionContext) -> Output:
        """
        Run inference on test dataset using trained RL agent.
        """
        logger = get_dagster_logger()

        import numpy as np
        import torch
        import torch.nn.functional as F

        from metapathpredict.config import Settings
        from metapathpredict.data import HDF5SequenceDataset
        from metapathpredict.models import (
            ActorCriticAgent,
            DQNAgent,
            PolicyGradientAgent,
        )
        from torch.utils.data import DataLoader

        settings = Settings()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load RL agent from checkpoint
        rl_path = settings.paths.weights_dir / "rl_best.pt"
        if not rl_path.exists():
            raise FileNotFoundError(f"RL checkpoint not found: {rl_path}")

        ckpt = torch.load(rl_path, map_location=device, weights_only=False)
        algorithm = ckpt.get("algorithm", "actor_critic")
        config = ckpt.get("config", {})

        agent_cls = {
            "dqn": DQNAgent,
            "policy_gradient": PolicyGradientAgent,
            "actor_critic": ActorCriticAgent,
        }.get(algorithm, ActorCriticAgent)

        agent = agent_cls(
            in_channels=4, num_actions=3,
            backbone=config.get("backbone", "medium"),
            hidden_dim=config.get("hidden_dim", 256),
        )
        agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
        agent.to(device).eval()
        logger.info(f"Loaded {algorithm} agent from {rl_path}")

        # Load test data
        test_path = (
            settings.paths.data_dir / "datasets" / "unified"
            / f"encoded_test_{settings.data.default_fragment_size}.hdf5"
        )
        test_dataset = HDF5SequenceDataset(str(test_path))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Predict
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                x, labels = batch[0].to(device), batch[1]
                if algorithm == "dqn":
                    out = agent(x)
                    probs = F.softmax(out, dim=1)
                elif algorithm == "policy_gradient":
                    probs, _ = agent(x)
                else:
                    logits, _ = agent(x)
                    probs = F.softmax(logits, dim=1)

                preds = probs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_targets.append(labels.numpy())

        predictions = np.concatenate(all_preds)
        probabilities = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        accuracy = float((predictions == targets).mean())

        # Save predictions
        output_path = settings.paths.data_dir / "output" / "predictions" / "test_predictions.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(output_path, predictions=predictions, probabilities=probabilities, targets=targets)
        logger.info(f"Predictions saved to {output_path}")

        return Output(
            value=str(output_path),
            metadata={
                "num_predictions": MetadataValue.int(len(predictions)),
                "accuracy": MetadataValue.float(accuracy),
                "algorithm": MetadataValue.text(algorithm),
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
