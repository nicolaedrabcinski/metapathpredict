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
    """Train a model using the specified pipeline."""
    from metapathpredict.config import Settings
    from metapathpredict.data import SequenceDataModule

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
    logger.info(f"Config: {args.config or 'default'}")
    logger.info(f"Pipeline: {getattr(args, 'pipeline', 'full')}")

    logger.info("Loading datasets...")
    data_module = SequenceDataModule.from_config(settings)
    logger.info(
        f"Datasets loaded: train={len(data_module.train_dataset)}, "
        f"val={len(data_module.val_dataset)}, "
        f"test={len(data_module.test_dataset)}"
    )

    output_dir = Path(args.output or settings.paths.weights_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    pipeline = getattr(args, "pipeline", "full")

    if pipeline == "supervised":
        return _train_supervised(args, settings, device, data_module, output_dir)
    elif pipeline == "contrastive":
        return _train_contrastive(settings, device, data_module, output_dir)
    elif pipeline == "rl":
        return _train_rl(settings, device, data_module, output_dir)
    elif pipeline == "full":
        return _train_full_pipeline(args, settings, device, data_module, output_dir)
    else:
        logger.error(f"Unknown pipeline: {pipeline}")
        return 1


def _train_supervised(args, settings, device, data_module, output_dir) -> int:
    """Train a supervised CNN model (legacy)."""
    from metapathpredict.models import UnifiedClassifier, create_cnn_model
    from metapathpredict.training import (
        EarlyStopping,
        MetricsLogger,
        ModelCheckpoint,
        ProgressCallback,
        Trainer,
        get_scheduler,
    )

    model_name = getattr(args, "model", "unified")
    if model_name == "unified":
        model = UnifiedClassifier(
            seq_length=settings.data.default_fragment_size,
            num_classes=settings.model.num_classes,
        )
    else:
        model = create_cnn_model(
            model_type=model_name,
            in_channels=4,
            num_classes=settings.model.num_classes,
        )

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )

    scheduler = get_scheduler(
        name=settings.training.scheduler or "warmup_cosine",
        optimizer=optimizer,
        total_epochs=settings.training.num_epochs,
        warmup_epochs=settings.training.warmup_epochs,
    )

    callbacks = [
        ProgressCallback(show_metrics=["accuracy", "f1_macro"]),
        MetricsLogger(log_dir=output_dir),
        ModelCheckpoint(save_dir=output_dir, monitor="val_loss", save_top_k=3),
        EarlyStopping(monitor="val_loss", patience=settings.training.patience),
    ]

    class_weights = None
    if settings.training.use_class_weights:
        class_weights = data_module.get_class_weights(device)
        logger.info(f"Using computed class weights: {class_weights.tolist()}")

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

    trainer.fit(num_epochs=settings.training.num_epochs)
    trainer.save_checkpoint(output_dir / "final_model.pt")
    logger.info(f"Supervised training complete. Model saved to {output_dir}")
    return 0


def _train_contrastive(settings, device, data_module, output_dir) -> int:
    """Train contrastive learning encoder (SimCLR / SupCon)."""
    import time

    from metapathpredict.models import (
        ContrastiveAugmentation,
        ContrastiveEncoder,
        ContrastiveTrainer,
    )

    cfg = settings.contrastive

    logger.info("-" * 50)
    logger.info("Contrastive config:")
    logger.info(f"  Backbone:       {cfg.backbone}")
    logger.info(f"  Base channels:  {cfg.base_channels}")
    logger.info(f"  Projection dim: {cfg.projection_dim}")
    logger.info(f"  Hidden dim:     {cfg.hidden_dim}")
    logger.info(f"  Loss type:      {cfg.loss_type}")
    logger.info(f"  Temperature:    {cfg.temperature}")
    logger.info(f"  Mutation rate:  {cfg.mutation_rate}")
    logger.info(f"  Mask rate:      {cfg.mask_rate}")
    logger.info(f"  Epochs:         {cfg.num_epochs}")
    logger.info(f"  Learning rate:  {cfg.learning_rate}")
    logger.info(f"  Weight decay:   {cfg.weight_decay}")
    logger.info(f"  Batch size:     {cfg.batch_size}")
    logger.info("-" * 50)

    logger.info("Building ContrastiveEncoder...")
    encoder = ContrastiveEncoder(
        in_channels=4,
        backbone=cfg.backbone,
        projection_dim=cfg.projection_dim,
        hidden_dim=cfg.hidden_dim,
        base_channels=cfg.base_channels,
    )
    encoder.to(device)

    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"ContrastiveEncoder: {total_params:,} total params ({trainable_params:,} trainable)")

    logger.info("Building augmentation pipeline...")
    augmentation = ContrastiveAugmentation(
        mutation_rate=cfg.mutation_rate,
        mask_rate=cfg.mask_rate,
    )

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    logger.info(f"Optimizer: AdamW (lr={cfg.learning_rate}, wd={cfg.weight_decay})")

    trainer = ContrastiveTrainer(
        encoder=encoder,
        optimizer=optimizer,
        augmentation=augmentation,
        temperature=cfg.temperature,
        use_supervised=(cfg.loss_type == "supcon"),
        device=device,
    )

    from tqdm import tqdm

    train_loader = data_module.train_dataloader()
    num_batches = len(train_loader)
    logger.info(f"Train loader: {num_batches} batches (batch_size={train_loader.batch_size})")
    logger.info("Starting contrastive training...")

    best_loss = float("inf")
    t0_total = time.time()

    epoch_pbar = tqdm(range(cfg.num_epochs), desc="Contrastive Epochs", unit="epoch")
    for epoch in epoch_pbar:
        t0_epoch = time.time()
        loss = trainer.train_epoch(train_loader)
        elapsed = time.time() - t0_epoch

        improved = ""
        if loss < best_loss:
            best_loss = loss
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "config": cfg.model_dump(),
            }, output_dir / "contrastive_best.pt")
            improved = " [BEST - saved]"

        epoch_pbar.set_postfix(
            loss=f"{loss:.6f}",
            best=f"{best_loss:.6f}",
            time=f"{elapsed:.1f}s",
        )

        logger.info(
            f"[Contrastive] Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"Loss: {loss:.6f} | Best: {best_loss:.6f} | "
            f"Time: {elapsed:.1f}s{improved}"
        )

    torch.save({
        "epoch": cfg.num_epochs - 1,
        "encoder_state_dict": encoder.state_dict(),
        "config": cfg.model_dump(),
    }, output_dir / "contrastive_final.pt")

    # The contrastive objective never touches encoder.encoder.classifier (it only
    # optimizes the projection head), so that head is still randomly initialized
    # here. Left as-is, _predict_single/prediction_service would softmax random
    # weights and report a confident-looking but meaningless class. Fit it as a
    # linear probe on the frozen backbone so contrastive-only checkpoints produce
    # real predictions instead.
    logger.info("Fitting linear-probe classifier head on frozen embeddings...")
    import torch.nn.functional as F

    for p in encoder.encoder.parameters():
        p.requires_grad = False
    for p in encoder.encoder.classifier.parameters():
        p.requires_grad = True

    probe_optimizer = torch.optim.Adam(encoder.encoder.classifier.parameters(), lr=1e-3)
    probe_epochs = 3
    encoder.train()
    for probe_epoch in range(probe_epochs):
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            x, labels = batch[0].to(device), batch[1].to(device)
            logits = encoder.encoder(x)
            probe_loss = F.cross_entropy(logits, labels)

            probe_optimizer.zero_grad()
            probe_loss.backward()
            probe_optimizer.step()

            total_loss += probe_loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        logger.info(
            f"  Linear probe epoch {probe_epoch + 1}/{probe_epochs}: "
            f"loss={total_loss / num_batches:.4f}, acc={correct / total:.4f}"
        )

    for p in encoder.encoder.parameters():
        p.requires_grad = True
    encoder.eval()

    # Re-save both checkpoints with the now-trained classifier head.
    torch.save({
        "epoch": cfg.num_epochs - 1,
        "encoder_state_dict": encoder.state_dict(),
        "loss": best_loss,
        "config": cfg.model_dump(),
    }, output_dir / "contrastive_best.pt")
    torch.save({
        "epoch": cfg.num_epochs - 1,
        "encoder_state_dict": encoder.state_dict(),
        "config": cfg.model_dump(),
    }, output_dir / "contrastive_final.pt")

    total_time = time.time() - t0_total
    logger.info(f"Contrastive training complete in {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"  Best loss: {best_loss:.6f}")
    logger.info(f"  Best checkpoint: {output_dir / 'contrastive_best.pt'}")
    logger.info(f"  Final checkpoint: {output_dir / 'contrastive_final.pt'}")
    return 0


def _train_rl(settings, device, data_module, output_dir,
              encoder_checkpoint=None) -> int:
    """Train RL agent for sequence classification."""
    import time

    from metapathpredict.models import (
        ActorCriticAgent,
        DQNAgent,
        PolicyGradientAgent,
        RLTrainer,
        SequenceEnvironment,
    )

    cfg = settings.rl

    logger.info("-" * 50)
    logger.info("RL config:")
    logger.info(f"  Algorithm:           {cfg.algorithm}")
    logger.info(f"  Backbone:            {cfg.backbone}")
    logger.info(f"  Hidden dim:          {cfg.hidden_dim}")
    logger.info(f"  Batch size:          {cfg.batch_size}")
    logger.info(f"  Epochs:              {cfg.num_epochs}")
    logger.info(f"  Episodes/epoch:      {cfg.episodes_per_epoch}")
    logger.info(f"  Learning rate:       {cfg.learning_rate}")
    logger.info(f"  Gamma:               {cfg.gamma}")
    logger.info(f"  Rewards (C/I/U):     {cfg.reward_correct}/{cfg.reward_incorrect}/{cfg.reward_uncertain}")
    if cfg.algorithm == "dqn":
        logger.info(f"  Epsilon:             {cfg.epsilon_start} -> {cfg.epsilon_end} (decay {cfg.epsilon_decay_epochs} epochs)")
        logger.info(f"  Replay buffer:       {cfg.replay_buffer_size}")
        logger.info(f"  Target update freq:  {cfg.target_update_freq}")
    logger.info("-" * 50)

    # Extract all sequences/labels from DataModule into tensors
    logger.info("Loading training data into RL environment...")
    all_sequences = []
    all_labels = []
    for batch in data_module.train_dataloader():
        seqs, labs = batch[0], batch[1]
        all_sequences.append(seqs)
        all_labels.append(labs)

    all_sequences = torch.cat(all_sequences, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Class distribution
    unique, counts = torch.unique(all_labels, return_counts=True)
    class_names = ["bacteria", "eukaryotic", "virus"]
    logger.info(f"RL environment: {len(all_sequences)} sequences, shape={list(all_sequences.shape)}")
    for cls_idx, cnt in zip(unique.tolist(), counts.tolist()):
        name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
        logger.info(f"  {name}: {cnt} ({100*cnt/len(all_sequences):.1f}%)")

    # Create agent
    agent_map = {
        "dqn": DQNAgent,
        "policy_gradient": PolicyGradientAgent,
        "actor_critic": ActorCriticAgent,
    }
    AgentClass = agent_map[cfg.algorithm]
    logger.info(f"Building {AgentClass.__name__}...")
    agent = AgentClass(
        in_channels=4,
        num_actions=settings.model.num_classes,
        backbone=cfg.backbone,
        hidden_dim=cfg.hidden_dim,
        base_channels=settings.contrastive.base_channels,
    )
    agent.to(device)

    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    logger.info(f"{AgentClass.__name__}: {total_params:,} total params ({trainable_params:,} trainable)")

    # Transfer contrastive encoder weights if available
    ckpt_path = encoder_checkpoint or cfg.load_encoder_from
    if ckpt_path and Path(ckpt_path).exists():
        logger.info(f"Loading contrastive encoder from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder_sd = ckpt["encoder_state_dict"]
        # Filter to encoder.* keys only (skip projection head)
        agent_keys = {k for k, _ in agent.named_parameters()}
        transfer_sd = {k: v for k, v in encoder_sd.items() if k.startswith("encoder.") and k in agent_keys}
        missing, unexpected = agent.load_state_dict(transfer_sd, strict=False)
        logger.info(f"Weight transfer: {len(transfer_sd)} tensors transferred, {len(missing)} agent-specific (not transferred)")
        enc_epoch = ckpt.get("epoch")
        enc_loss = ckpt.get("loss")
        if enc_epoch is not None and enc_loss is not None:
            logger.info(f"  Encoder best checkpoint: epoch {enc_epoch + 1}, loss={enc_loss:.6f}")
    else:
        logger.info("No contrastive encoder checkpoint — training RL from scratch")

    # Create environment
    logger.info("Creating SequenceEnvironment...")
    env = SequenceEnvironment(
        sequences=all_sequences,
        labels=all_labels,
        reward_correct=cfg.reward_correct,
        reward_incorrect=cfg.reward_incorrect,
        reward_uncertain=cfg.reward_uncertain,
    )

    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    logger.info(f"Optimizer: AdamW (lr={cfg.learning_rate}, wd={cfg.weight_decay})")

    rl_trainer = RLTrainer(
        agent=agent,
        environment=env,
        optimizer=optimizer,
        device=device,
        gamma=cfg.gamma,
        algorithm=cfg.algorithm,
    )

    from tqdm import tqdm

    logger.info(f"Starting RL training: {cfg.num_epochs} epochs x {cfg.episodes_per_epoch} episodes...")
    best_accuracy = 0.0
    t0_total = time.time()

    epoch_pbar = tqdm(range(cfg.num_epochs), desc="RL Training", unit="epoch")
    for epoch in epoch_pbar:
        t0_epoch = time.time()

        # Epsilon decay for DQN
        if cfg.algorithm == "dqn":
            frac = min(epoch / max(cfg.epsilon_decay_epochs, 1), 1.0)
            epsilon = cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)
        else:
            epsilon = 0.0

        metrics = rl_trainer.train_epoch(
            num_episodes=cfg.episodes_per_epoch,
            epsilon=epsilon,
            batch_size=cfg.batch_size,
        )
        elapsed = time.time() - t0_epoch

        # Update target network for DQN
        if cfg.algorithm == "dqn" and (epoch + 1) % cfg.target_update_freq == 0:
            rl_trainer.update_target_network()
            logger.info(f"  Target network updated (epoch {epoch + 1})")

        improved = ""
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": cfg.model_dump(),
                "algorithm": cfg.algorithm,
                "base_channels": settings.contrastive.base_channels,
            }, output_dir / "rl_best.pt")
            improved = " [BEST - saved]"

        # Update tqdm
        epoch_pbar.set_postfix(
            acc=f"{metrics['accuracy']:.4f}",
            rew=f"{metrics['avg_reward']:.3f}",
            loss=f"{metrics['avg_loss']:.4f}",
            best=f"{best_accuracy:.4f}",
        )

        eps_str = f" | Eps: {epsilon:.3f}" if cfg.algorithm == "dqn" else ""
        logger.info(
            f"[RL] Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"Reward: {metrics['avg_reward']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Loss: {metrics['avg_loss']:.4f}{eps_str} | "
            f"Time: {elapsed:.1f}s{improved}"
        )

    torch.save({
        "epoch": cfg.num_epochs - 1,
        "agent_state_dict": agent.state_dict(),
        "config": cfg.model_dump(),
        "algorithm": cfg.algorithm,
        "base_channels": settings.contrastive.base_channels,
    }, output_dir / "rl_final.pt")

    total_time = time.time() - t0_total
    logger.info(f"RL training complete in {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"  Best accuracy: {best_accuracy:.4f}")
    logger.info(f"  Best checkpoint: {output_dir / 'rl_best.pt'}")
    logger.info(f"  Final checkpoint: {output_dir / 'rl_final.pt'}")
    return 0


def _train_full_pipeline(args, settings, device, data_module, output_dir) -> int:
    """Full pipeline: contrastive pretrain -> RL fine-tune."""
    import time

    t0 = time.time()

    logger.info("=" * 60)
    logger.info("  FULL PIPELINE: Contrastive Pretrain -> RL Fine-tune")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    logger.info("=" * 60)
    logger.info("  Phase 1/2: Contrastive Pretraining")
    logger.info("=" * 60)
    rc = _train_contrastive(settings, device, data_module, output_dir)
    if rc != 0:
        logger.error("Contrastive pretraining failed!")
        return rc

    contrastive_ckpt = output_dir / "contrastive_best.pt"
    logger.info("")

    logger.info("=" * 60)
    logger.info("  Phase 2/2: RL Fine-tuning (with contrastive encoder)")
    logger.info("=" * 60)
    rc = _train_rl(settings, device, data_module, output_dir,
                   encoder_checkpoint=contrastive_ckpt)
    if rc != 0:
        logger.error("RL fine-tuning failed!")
        return rc

    total_time = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("  FULL PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total time:            {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"  Contrastive checkpoint: {contrastive_ckpt}")
    logger.info(f"  RL checkpoint:          {output_dir / 'rl_best.pt'}")
    logger.info("=" * 60)
    return 0


def _read_fasta(fasta_path: Path) -> tuple[list[str], list[str]]:
    """Read sequences and IDs from a FASTA file."""
    sequences = []
    seq_ids = []
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
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

    return sequences, seq_ids


def _encode_sequence_tensor(sequence: str, fragment_size: int = 500) -> torch.Tensor:
    """One-hot encode a DNA sequence to tensor [1, 4, fragment_size]."""
    nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = sequence.upper()[:fragment_size]
    tensor = torch.zeros(1, 4, fragment_size)
    for i, nuc in enumerate(seq):
        idx = nuc_to_idx.get(nuc)
        if idx is not None:
            tensor[0, idx, i] = 1.0
    return tensor


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load contrastive or RL model from a checkpoint file.

    Returns (model, model_type) where model_type is 'contrastive' or 'rl'.
    """
    import torch.nn.functional as F
    from metapathpredict.models import (
        ActorCriticAgent,
        ContrastiveEncoder,
        DQNAgent,
        PolicyGradientAgent,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    if "encoder_state_dict" in ckpt:
        # Contrastive model
        model = ContrastiveEncoder(
            in_channels=4,
            backbone=config.get("backbone", "medium"),
            projection_dim=config.get("projection_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            base_channels=config.get("base_channels", 64),
        )
        model.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        model.to(device).eval()
        return model, "contrastive"

    elif "agent_state_dict" in ckpt:
        # RL model
        algorithm = ckpt.get("algorithm", "actor_critic")
        agent_cls = {
            "dqn": DQNAgent,
            "policy_gradient": PolicyGradientAgent,
            "actor_critic": ActorCriticAgent,
        }.get(algorithm, ActorCriticAgent)

        model = agent_cls(
            in_channels=4,
            num_actions=3,
            backbone=config.get("backbone", "medium"),
            hidden_dim=config.get("hidden_dim", 256),
            base_channels=ckpt.get("base_channels", 64),
        )
        model.load_state_dict(ckpt["agent_state_dict"], strict=False)
        model.to(device).eval()
        return model, "rl", algorithm

    else:
        raise ValueError(f"Unknown checkpoint format: {list(ckpt.keys())}")


def _predict_single(model, model_type: str, x: torch.Tensor,
                     algorithm: str = "actor_critic") -> tuple[int, float, list[float]]:
    """Run single prediction, return (class_idx, confidence, probs)."""
    import torch.nn.functional as F

    with torch.no_grad():
        if model_type == "contrastive":
            # Try classifier head on encoder
            output = model.encoder(x)
            if output.shape[-1] == 3:
                probs = F.softmax(output, dim=1).squeeze(0)
            else:
                emb = model.get_embeddings(x)
                if hasattr(model.encoder, "classifier"):
                    logits = model.encoder.classifier(emb)
                    probs = F.softmax(logits, dim=1).squeeze(0)
                else:
                    probs = torch.tensor([0.34, 0.33, 0.33])
        else:
            # RL model
            if algorithm == "dqn":
                q_values = model(x)
                probs = F.softmax(q_values, dim=1).squeeze(0)
            elif algorithm == "policy_gradient":
                action_probs, _ = model(x)
                probs = action_probs.squeeze(0)
            else:  # actor_critic
                logits, _ = model(x)
                probs = F.softmax(logits, dim=1).squeeze(0)

    class_idx = probs.argmax().item()
    confidence = probs[class_idx].item()
    return class_idx, confidence, probs.tolist()


def predict_command(args: argparse.Namespace) -> int:
    """Run inference on input data using contrastive/RL models."""
    from metapathpredict.config import Settings

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

    result = _load_model_from_checkpoint(checkpoint_path, device)
    if len(result) == 3:
        model, model_type, algorithm = result
    else:
        model, model_type = result
        algorithm = "actor_critic"

    logger.info(f"Loaded {model_type} model from {checkpoint_path}")

    # Process input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if input_path.suffix not in (".fasta", ".fa", ".fna"):
        logger.error(f"Unsupported input format: {input_path.suffix}")
        return 1

    sequences, seq_ids = _read_fasta(input_path)
    logger.info(f"Loaded {len(sequences)} sequences from {input_path}")

    fragment_size = settings.data.default_fragment_size
    class_names = ["bacteria", "eukaryotic", "virus"]

    # Make predictions
    output_path = Path(args.output) if args.output else input_path.with_suffix(".predictions.tsv")

    with open(output_path, "w") as f:
        f.write("sequence_id\tpredicted_class\tconfidence\t")
        f.write("\t".join([f"prob_{c}" for c in class_names]))
        f.write("\n")

        class_counts = {c: 0 for c in class_names}
        total_conf = 0.0

        for i, (seq_id, seq) in enumerate(zip(seq_ids, sequences)):
            x = _encode_sequence_tensor(seq, fragment_size).to(device)
            class_idx, confidence, probs = _predict_single(
                model, model_type, x, algorithm
            )
            pred_label = class_names[class_idx]
            class_counts[pred_label] += 1
            total_conf += confidence

            f.write(f"{seq_id}\t{pred_label}\t{confidence:.4f}\t")
            f.write("\t".join([f"{p:.4f}" for p in probs]))
            f.write("\n")

            if (i + 1) % 1000 == 0:
                logger.info(f"  Predicted {i + 1}/{len(sequences)}...")

    logger.info(f"Predictions saved to {output_path}")

    print("\nPrediction Summary:")
    print("-" * 40)
    for c in class_names:
        count = class_counts[c]
        pct = 100 * count / len(sequences) if sequences else 0
        print(f"  {c}: {count} ({pct:.1f}%)")
    print(f"  Average confidence: {total_conf / max(len(sequences), 1):.4f}")

    return 0


def _stream_fasta(fasta_path: Path):
    """Yield sequences one at a time from a FASTA file (streaming, low memory)."""
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    yield "".join(current_seq)
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            yield "".join(current_seq)


def prepare_command(args: argparse.Namespace) -> int:
    """Prepare dataset from FASTA files (streaming, supports millions of fragments)."""
    from metapathpredict.config import Settings
    from metapathpredict.data import OneHotEncoder, SequencePreprocessor

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

    # Create preprocessor and encoder
    preprocessor = SequencePreprocessor()
    encoder = OneHotEncoder()

    # Process each input file
    class_mapping = {
        "bacteria": 0,
        "eukaryotic": 1,
        "eucaryotic": 1,
        "fungi": 1,
        "fungal": 1,
        "protozoa": 1,
        "virus": 2,
        "viruses": 2,
        "viral": 2,
    }

    max_per_class = args.max_fragments
    chunk_size = args.chunk_size

    # ── Pass 1: stream fragments directly into a temporary HDF5 ──
    tmp_path = output_dir / f"_tmp_all_{sequence_length}.hdf5"
    total_written = 0
    class_counts = {0: 0, 1: 0, 2: 0}

    with h5py.File(tmp_path, "w") as hf:
        ds_seq = hf.create_dataset(
            "sequences",
            shape=(0, 4, sequence_length),
            maxshape=(None, 4, sequence_length),
            dtype="float32",
            chunks=(min(chunk_size, 1024), 4, sequence_length),
            compression="gzip",
            compression_opts=1,
        )
        ds_lab = hf.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            dtype="int64",
            chunks=(min(chunk_size, 4096),),
            compression="gzip",
            compression_opts=1,
        )

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

            # Stream sequences, fragment, encode, and write in chunks
            buf_data = []
            buf_labels = []
            seq_count = 0
            class_written = class_counts[class_label]

            for seq in _stream_fasta(fasta_path):
                seq_count += 1
                if max_per_class and class_written >= max_per_class:
                    break
                try:
                    cleaned, stats = preprocessor.process(seq)
                    if cleaned is None:
                        continue
                    for fragment, _start, _end in preprocessor.fragment(
                        cleaned, sequence_length, step_size=sequence_length
                    ):
                        encoded = encoder.encode(fragment)  # (seq_len, 4)
                        # Transpose to (4, seq_len) for Conv1d
                        buf_data.append(encoded.T)
                        buf_labels.append(class_label)
                        class_written += 1

                        if max_per_class and class_written >= max_per_class:
                            break

                        # Flush chunk to HDF5 when buffer is full
                        if len(buf_data) >= chunk_size:
                            chunk_arr = np.stack(buf_data, axis=0).astype(np.float32)
                            chunk_lab = np.array(buf_labels, dtype=np.int64)
                            n = len(chunk_arr)
                            ds_seq.resize(total_written + n, axis=0)
                            ds_lab.resize(total_written + n, axis=0)
                            ds_seq[total_written : total_written + n] = chunk_arr
                            ds_lab[total_written : total_written + n] = chunk_lab
                            total_written += n
                            buf_data.clear()
                            buf_labels.clear()

                except Exception as e:
                    logger.warning(f"  Failed to encode sequence: {e}")

            # Flush remaining buffer
            if buf_data:
                chunk_arr = np.stack(buf_data, axis=0).astype(np.float32)
                chunk_lab = np.array(buf_labels, dtype=np.int64)
                n = len(chunk_arr)
                ds_seq.resize(total_written + n, axis=0)
                ds_lab.resize(total_written + n, axis=0)
                ds_seq[total_written : total_written + n] = chunk_arr
                ds_lab[total_written : total_written + n] = chunk_lab
                total_written += n
                buf_data.clear()
                buf_labels.clear()

            class_counts[class_label] = class_written
            logger.info(f"  Processed {seq_count} sequences → {class_written} fragments")

    if total_written == 0:
        logger.error("No sequences processed")
        tmp_path.unlink(missing_ok=True)
        return 1

    logger.info(f"Total: {total_written} fragments written to temp HDF5")
    canonical_names = {0: "bacteria", 1: "eukaryotic", 2: "virus"}
    for label, name in canonical_names.items():
        if class_counts[label] > 0:
            logger.info(f"  {name}: {class_counts[label]}")

    # ── Pass 2: shuffle and split into train/val/test ──
    logger.info("Shuffling and splitting into train/val/test...")

    rng = np.random.RandomState(42)
    indices = rng.permutation(total_written)

    n_test = int(total_written * 0.1)
    n_val = int(total_written * 0.1)
    n_train = total_written - n_val - n_test

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    with h5py.File(tmp_path, "r") as src:
        for split_name, split_idx in splits.items():
            split_idx_sorted = np.sort(split_idx)  # HDF5 needs sorted indices
            out_path = output_dir / f"encoded_{split_name}_{sequence_length}.hdf5"

            logger.info(f"Writing {split_name}: {len(split_idx)} samples → {out_path}")

            with h5py.File(out_path, "w") as dst:
                # Read and write in chunks to avoid OOM
                n = len(split_idx_sorted)
                read_chunk = min(chunk_size, n)
                seq_ds = dst.create_dataset(
                    "sequences",
                    shape=(n, 4, sequence_length),
                    dtype="float32",
                    chunks=(min(1024, n), 4, sequence_length),
                    compression="gzip",
                )
                lab_ds = dst.create_dataset(
                    "labels",
                    shape=(n,),
                    dtype="int64",
                    chunks=(min(4096, n),),
                    compression="gzip",
                )

                written = 0
                for start in range(0, n, read_chunk):
                    end = min(start + read_chunk, n)
                    idx_batch = split_idx_sorted[start:end]
                    seq_ds[written : written + len(idx_batch)] = src["sequences"][idx_batch]
                    lab_ds[written : written + len(idx_batch)] = src["labels"][idx_batch]
                    written += len(idx_batch)

    # Remove temp file
    tmp_path.unlink(missing_ok=True)

    # Save metadata
    metadata = {
        "sequence_length": sequence_length,
        "num_classes": 3,
        "class_names": ["bacteria", "eukaryotic", "virus"],
        "total_fragments": total_written,
        "train_size": n_train,
        "val_size": n_val,
        "test_size": n_test,
        "class_counts": {canonical_names[k]: v for k, v in class_counts.items()},
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset prepared in {output_dir}")
    logger.info(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    """Evaluate contrastive/RL model on test data."""
    from metapathpredict.config import Settings
    from metapathpredict.data import HDF5SequenceDataset

    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader

    if args.config:
        settings = Settings.from_yaml(args.config)
    else:
        settings = Settings()

    device = setup_device(args)

    # Load model
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        logger.error(f"Model checkpoint not found: {checkpoint_path}")
        return 1

    result = _load_model_from_checkpoint(checkpoint_path, device)
    if len(result) == 3:
        model, model_type, algorithm = result
    else:
        model, model_type = result
        algorithm = "actor_critic"

    logger.info(f"Evaluating {model_type} model from {checkpoint_path}")

    # Load test data
    test_dataset = HDF5SequenceDataset(args.data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size or 64,
        shuffle=False,
        num_workers=4,
    )

    class_names = ["bacteria", "eukaryotic", "virus"]
    all_preds = []
    all_targets = []

    for batch in test_loader:
        seqs, labels = batch[0].to(device), batch[1]
        for i in range(seqs.shape[0]):
            x = seqs[i].unsqueeze(0)
            class_idx, _, _ = _predict_single(model, model_type, x, algorithm)
            all_preds.append(class_idx)
            all_targets.append(labels[i].item())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = (all_preds == all_targets).mean()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)

    if args.output:
        output_path = Path(args.output)
        eval_results = {
            "accuracy": float(accuracy),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                all_targets, all_preds,
                target_names=class_names,
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
    train_parser.add_argument("--pipeline", "-p", default="full",
                             choices=["full", "contrastive", "rl", "supervised"],
                             help="Training pipeline (default: full = contrastive + RL)")
    train_parser.add_argument("--model", "-m", default="unified",
                             choices=["unified", "simple", "multiscale", "residual"],
                             help="Model architecture (only for supervised pipeline)")
    train_parser.add_argument("--epochs", "-e", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", "-lr", type=float, help="Learning rate")
    train_parser.add_argument("--output", "-o", help="Output directory")
    train_parser.add_argument("--device", help="Device (cuda/cpu)")
    train_parser.set_defaults(func=train_command)
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--model", "-m", required=True,
                                help="Path to model checkpoint (contrastive_best.pt or rl_best.pt)")
    predict_parser.add_argument("--input", "-i", required=True, help="Input FASTA file")
    predict_parser.add_argument("--output", "-o", help="Output file")
    predict_parser.add_argument("--config", "-c", help="Path to config file")
    predict_parser.add_argument("--device", help="Device (cuda/cpu)")
    predict_parser.set_defaults(func=predict_command)
    
    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prepare_parser.add_argument("inputs", nargs="+", help="Input FASTA files")
    prepare_parser.add_argument("--output", "-o", required=True, help="Output directory")
    prepare_parser.add_argument("--config", "-c", help="Path to config file")
    prepare_parser.add_argument("--length", "-l", type=int, help="Sequence length")
    prepare_parser.add_argument("--max-fragments", type=int, default=None,
                                help="Max fragments per class (default: unlimited)")
    prepare_parser.add_argument("--chunk-size", type=int, default=10000,
                                help="Write chunk size (controls memory usage, default: 10000)")
    prepare_parser.set_defaults(func=prepare_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--model", "-m", required=True,
                             help="Path to model checkpoint (contrastive_best.pt or rl_best.pt)")
    eval_parser.add_argument("--data", "-d", required=True, help="Test data HDF5 file")
    eval_parser.add_argument("--output", "-o", help="Output file for results")
    eval_parser.add_argument("--config", "-c", help="Path to config file")
    eval_parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
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
