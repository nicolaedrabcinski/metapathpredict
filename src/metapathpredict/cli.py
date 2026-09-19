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


class EarlyStopping:
    """
    Stops a training loop once a validation metric hasn't improved for
    `patience` epochs, rather than always running the configured epoch count.
    """

    def __init__(self, patience: int, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        """Record this epoch's value; return True if training should stop now."""
        if self.patience <= 0:
            return False
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def _cap_threads(settings, max_threads: int) -> None:
    """Cap torch CPU threads and DataLoader workers (call before building the data module)."""
    torch.set_num_threads(max_threads)
    if settings.data.num_workers > max_threads:
        logger.info(f"Capping data.num_workers {settings.data.num_workers} -> {max_threads}")
        settings.data.num_workers = max_threads


def _apply_dataset_labels(settings, data_module) -> None:
    """The prepared dataset (HDF5 attrs written by `prepare`) decides the label space."""
    settings.model.num_classes = getattr(data_module.train_dataset, "num_classes", 3)
    settings.data.class_names = list(
        getattr(data_module.train_dataset, "class_names", settings.data.class_names)
    )
    logger.info(f"Classes ({settings.model.num_classes}): {settings.data.class_names}")


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

    # Cap CPU thread usage (torch intra-op parallelism + DataLoader workers)
    # regardless of how many cores are available on the box.
    _cap_threads(settings, getattr(args, "max_threads", 16))

    device = setup_device(args)
    logger.info(f"Training on {device}")
    logger.info(f"Config: {args.config or 'default'}")
    logger.info(f"Pipeline: {getattr(args, 'pipeline', 'full')}")

    logger.info("Loading datasets...")
    data_module = SequenceDataModule.from_config(settings)
    _apply_dataset_labels(settings, data_module)
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
        num_classes=settings.model.num_classes,
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

    # contrastive.batch_size sets how many negatives each anchor sees (2*batch - 2 with
    # NT-Xent). It used to be ignored: both phases silently used training.batch_size.
    train_loader = data_module.train_dataloader(batch_size=cfg.batch_size)
    val_loader = data_module.val_dataloader(batch_size=cfg.batch_size)
    num_batches = len(train_loader)
    logger.info(f"Train loader: {num_batches} batches (batch_size={train_loader.batch_size})")
    logger.info("Starting contrastive training...")

    # Track the checkpoint by val loss, not train loss — a model can keep
    # driving train loss down on data it's memorizing while val loss flattens
    # or rises; that gap is the standard overfitting signal.
    best_val_loss = float("inf")
    t0_total = time.time()
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="min")
    last_epoch = 0

    epoch_pbar = tqdm(range(cfg.num_epochs), desc="Contrastive Epochs", unit="epoch")
    for epoch in epoch_pbar:
        last_epoch = epoch
        t0_epoch = time.time()
        loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate_epoch(val_loader)
        elapsed = time.time() - t0_epoch

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "val_loss": val_loss,
                "config": cfg.model_dump(),
                "num_classes": settings.model.num_classes,
                "class_names": settings.data.class_names,
            }, output_dir / "contrastive_best.pt")
            improved = " [BEST - saved]"

        epoch_pbar.set_postfix(
            loss=f"{loss:.6f}",
            val_loss=f"{val_loss:.6f}",
            best_val=f"{best_val_loss:.6f}",
            time=f"{elapsed:.1f}s",
        )

        logger.info(
            f"[Contrastive] Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"Loss: {loss:.6f} | Val loss: {val_loss:.6f} | Best val: {best_val_loss:.6f} | "
            f"Time: {elapsed:.1f}s{improved}"
        )

        if early_stopper.step(val_loss):
            logger.info(
                f"Early stopping: val loss hasn't improved for "
                f"{cfg.early_stopping_patience} epochs, stopping at epoch {epoch + 1}/{cfg.num_epochs}"
            )
            break

    torch.save({
        "epoch": last_epoch,
        "encoder_state_dict": encoder.state_dict(),
        "config": cfg.model_dump(),
        "num_classes": settings.model.num_classes,
        "class_names": settings.data.class_names,
    }, output_dir / "contrastive_final.pt")

    # The contrastive objective never touches encoder.encoder.classifier (it only
    # optimizes the projection head), so that head is still randomly initialized
    # here. Left as-is, _predict_single/prediction_service would softmax random
    # weights and report a confident-looking but meaningless class. Fit it as a
    # linear probe on the frozen backbone so contrastive-only checkpoints produce
    # real predictions instead.
    logger.info("Fitting linear-probe classifier head on frozen embeddings...")
    import copy

    import torch.nn.functional as F

    for p in encoder.encoder.parameters():
        p.requires_grad = False
    for p in encoder.encoder.classifier.parameters():
        p.requires_grad = True

    probe_optimizer = torch.optim.Adam(encoder.encoder.classifier.parameters(), lr=1e-3)
    probe_epochs = cfg.probe_epochs
    best_probe_val_acc = -1.0
    best_probe_state = None
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

        encoder.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, labels = batch[0].to(device), batch[1].to(device)
                logits = encoder.encoder(x)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        encoder.train()

        logger.info(
            f"  Linear probe epoch {probe_epoch + 1}/{probe_epochs}: "
            f"train_loss={total_loss / num_batches:.4f}, train_acc={correct / total:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        # Keep the classifier weights from whichever epoch generalized best,
        # same reasoning as picking the encoder checkpoint by val loss above.
        if val_acc > best_probe_val_acc:
            best_probe_val_acc = val_acc
            best_probe_state = copy.deepcopy(encoder.encoder.classifier.state_dict())

    for p in encoder.encoder.parameters():
        p.requires_grad = True
    if best_probe_state is not None:
        encoder.encoder.classifier.load_state_dict(best_probe_state)
    encoder.eval()

    # Re-save both checkpoints with the now-trained classifier head.
    torch.save({
        "epoch": last_epoch,
        "encoder_state_dict": encoder.state_dict(),
        "val_loss": best_val_loss,
        "probe_val_acc": best_probe_val_acc,
        "config": cfg.model_dump(),
        "num_classes": settings.model.num_classes,
        "class_names": settings.data.class_names,
    }, output_dir / "contrastive_best.pt")
    torch.save({
        "epoch": last_epoch,
        "encoder_state_dict": encoder.state_dict(),
        "config": cfg.model_dump(),
        "num_classes": settings.model.num_classes,
        "class_names": settings.data.class_names,
    }, output_dir / "contrastive_final.pt")

    total_time = time.time() - t0_total
    logger.info(f"Contrastive training complete in {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"  Best val loss: {best_val_loss:.6f}")
    logger.info(f"  Best probe val accuracy: {best_probe_val_acc:.4f}")
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

    # Point the RL environment at the train dataset directly instead of
    # materializing it into one CPU tensor: at 8KB/fragment (float32 one-hot,
    # 4x500), the full unified_v2 split (8.5M fragments) would need ~68GB of
    # RAM. SequenceEnvironment reads samples lazily via dataset[idx] instead.
    logger.info("Loading training data into RL environment...")
    train_dataset = data_module.train_dataset
    logger.info(f"RL environment: {len(train_dataset)} sequences")

    # Class distribution. Read the labels array directly when backed by HDF5
    # (like HDF5SequenceDataset.get_class_weights does) instead of paying for
    # a full sequence read per sample just to look at its label.
    class_names = settings.data.class_names
    if hasattr(train_dataset, "hdf5_path") and hasattr(train_dataset, "labels_key"):
        import h5py
        with h5py.File(train_dataset.hdf5_path, "r") as f:
            all_labels = f[train_dataset.labels_key][:]
        unique, label_counts = np.unique(all_labels, return_counts=True)
        counts = dict(zip(unique.tolist(), label_counts.tolist()))
    elif hasattr(train_dataset, "labels") and torch.is_tensor(train_dataset.labels):
        unique, label_counts = np.unique(train_dataset.labels.numpy(), return_counts=True)
        counts = dict(zip(unique.tolist(), label_counts.tolist()))
    else:
        counts: dict[int, int] = {}
        for _, label in train_dataset:
            label_idx = label.item() if torch.is_tensor(label) else int(label)
            counts[label_idx] = counts.get(label_idx, 0) + 1
    for cls_idx in sorted(counts):
        name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
        logger.info(f"  {name}: {counts[cls_idx]} ({100*counts[cls_idx]/len(train_dataset):.1f}%)")

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
        enc_val_loss = ckpt.get("val_loss")
        if enc_epoch is not None and enc_val_loss is not None:
            logger.info(f"  Encoder best checkpoint: epoch {enc_epoch + 1}, val_loss={enc_val_loss:.6f}")
    else:
        logger.info("No contrastive encoder checkpoint — training RL from scratch")

    # Create environment
    logger.info("Creating SequenceEnvironment...")
    env = SequenceEnvironment(
        sequences=train_dataset,
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

    val_dataset = data_module.val_dataset

    def evaluate_on_val() -> float:
        """
        Deterministic accuracy on the val split — no exploration, no gradient.
        Used to pick the checkpoint that generalizes instead of the train-episode
        accuracy, which is measured *while the agent is still updating* within
        the epoch and mixes early (worse) and late (better) episodes together.
        """
        agent.eval()
        correct = 0
        eval_batch = 256
        with torch.no_grad():
            for start in range(0, len(val_dataset), eval_batch):
                items = [val_dataset[i] for i in range(start, min(start + eval_batch, len(val_dataset)))]
                x = torch.stack([it[0] for it in items]).to(device)
                y = torch.stack([
                    it[1] if torch.is_tensor(it[1]) else torch.tensor(it[1]) for it in items
                ]).to(device)

                if cfg.algorithm == "dqn":
                    logits = agent(x)
                elif cfg.algorithm == "policy_gradient":
                    logits, _ = agent(x)
                else:
                    logits, _ = agent(x)
                correct += (logits.argmax(dim=1) == y).sum().item()
        agent.train()
        return correct / len(val_dataset)

    logger.info(f"Starting RL training: {cfg.num_epochs} epochs x {cfg.episodes_per_epoch} episodes...")
    best_val_accuracy = 0.0
    t0_total = time.time()
    early_stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="max")
    last_epoch = 0

    epoch_pbar = tqdm(range(cfg.num_epochs), desc="RL Training", unit="epoch")
    for epoch in epoch_pbar:
        last_epoch = epoch
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
        val_accuracy = evaluate_on_val()
        elapsed = time.time() - t0_epoch

        # Update target network for DQN
        if cfg.algorithm == "dqn" and (epoch + 1) % cfg.target_update_freq == 0:
            rl_trainer.update_target_network()
            logger.info(f"  Target network updated (epoch {epoch + 1})")

        improved = ""
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "epoch": epoch,
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "val_accuracy": val_accuracy,
                "config": cfg.model_dump(),
                "algorithm": cfg.algorithm,
                "base_channels": settings.contrastive.base_channels,
                "num_classes": settings.model.num_classes,
                "class_names": settings.data.class_names,
            }, output_dir / "rl_best.pt")
            improved = " [BEST - saved]"

        # Update tqdm
        epoch_pbar.set_postfix(
            acc=f"{metrics['accuracy']:.4f}",
            val_acc=f"{val_accuracy:.4f}",
            rew=f"{metrics['avg_reward']:.3f}",
            loss=f"{metrics['avg_loss']:.4f}",
            best_val=f"{best_val_accuracy:.4f}",
        )

        eps_str = f" | Eps: {epsilon:.3f}" if cfg.algorithm == "dqn" else ""
        logger.info(
            f"[RL] Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"Reward: {metrics['avg_reward']:.4f} | "
            f"Train acc: {metrics['accuracy']:.4f} | Val acc: {val_accuracy:.4f} | "
            f"Loss: {metrics['avg_loss']:.4f}{eps_str} | "
            f"Time: {elapsed:.1f}s{improved}"
        )

        if early_stopper.step(val_accuracy):
            logger.info(
                f"Early stopping: val accuracy hasn't improved for "
                f"{cfg.early_stopping_patience} epochs, stopping at epoch {epoch + 1}/{cfg.num_epochs}"
            )
            break

    torch.save({
        "epoch": last_epoch,
        "agent_state_dict": agent.state_dict(),
        "config": cfg.model_dump(),
        "algorithm": cfg.algorithm,
        "base_channels": settings.contrastive.base_channels,
        "num_classes": settings.model.num_classes,
        "class_names": settings.data.class_names,
    }, output_dir / "rl_final.pt")

    total_time = time.time() - t0_total
    logger.info(f"RL training complete in {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"  Best val accuracy: {best_val_accuracy:.4f}")
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
            num_classes=ckpt.get("num_classes", 3),
        )
        model.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        model.class_names = ckpt.get("class_names", ["bacteria", "eukaryotic", "virus"])
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
            num_actions=ckpt.get("num_classes", 3),
            backbone=config.get("backbone", "medium"),
            hidden_dim=config.get("hidden_dim", 256),
            base_channels=ckpt.get("base_channels", 64),
        )
        model.load_state_dict(ckpt["agent_state_dict"], strict=False)
        model.class_names = ckpt.get("class_names", ["bacteria", "eukaryotic", "virus"])
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
            # Classifier head fit as a linear probe at the end of contrastive training
            probs = F.softmax(model.encoder(x), dim=1).squeeze(0)
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
    class_names = list(getattr(model, "class_names", ["bacteria", "eukaryotic", "virus"]))

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
    import gzip

    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    current_seq = []
    with opener(fasta_path, "rt") as f:
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


def _assign_sequence_splits(
    n_sequences: int, train_ratio: float, val_ratio: float, test_ratio: float, rng: np.random.RandomState,
) -> list[str]:
    """
    Assign each of n_sequences source sequences to train/val/test, so a whole
    sequence (and every fragment cut from it) lands entirely in one split.

    A fragment-level split lets fragments from the same genome appear in both
    train and val — the model can then pick up on that genome's specific
    composition rather than learning something that generalizes, which makes
    val accuracy an overly optimistic estimate of real generalization.
    """
    order = rng.permutation(n_sequences)
    if n_sequences <= 1:
        return ["train"] * n_sequences
    if n_sequences == 2:
        # Not enough sequences to give every split at least one; prefer train+val.
        splits = [""] * n_sequences
        splits[order[0]] = "train"
        splits[order[1]] = "val"
        return splits

    n_val = max(1, round(n_sequences * val_ratio))
    n_test = max(1, round(n_sequences * test_ratio))
    n_train = n_sequences - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = max(1, (n_sequences - n_train) // 2)
        n_test = n_sequences - n_train - n_val

    splits = [""] * n_sequences
    for idx in order[:n_train]:
        splits[idx] = "train"
    for idx in order[n_train : n_train + n_val]:
        splits[idx] = "val"
    for idx in order[n_train + n_val :]:
        splits[idx] = "test"
    return splits


class _SplitWriter:
    """Growable HDF5 (sequences, labels) file with a chunk buffer."""

    def __init__(self, path: Path, sequence_length: int, chunk_size: int, attrs: dict):
        import h5py

        self.file = h5py.File(path, "w")
        self.seq = self.file.create_dataset(
            "sequences", shape=(0, 4, sequence_length), maxshape=(None, 4, sequence_length),
            dtype="float32", chunks=(min(chunk_size, 1024), 4, sequence_length),
            compression="gzip", compression_opts=1,
        )
        self.lab = self.file.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype="int64",
            chunks=(min(chunk_size, 4096),), compression="gzip", compression_opts=1,
        )
        for key, value in attrs.items():
            self.file.attrs[key] = value
        self.chunk_size = chunk_size
        self.written = 0
        self._data: list = []
        self._labels: list = []

    def append(self, arr: np.ndarray, label: int) -> None:
        self._data.append(arr)
        self._labels.append(label)
        if len(self._data) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._data:
            return
        chunk = np.stack(self._data, axis=0).astype(np.float32)
        n = len(chunk)
        self.seq.resize(self.written + n, axis=0)
        self.lab.resize(self.written + n, axis=0)
        self.seq[self.written : self.written + n] = chunk
        self.lab[self.written : self.written + n] = np.array(self._labels, dtype=np.int64)
        self.written += n
        self._data.clear()
        self._labels.clear()

    def close(self) -> None:
        self.flush()
        self.file.close()


def _allocate_quotas(capacities: list[float], total: float) -> list[float]:
    """Split `total` across genomes as evenly as possible without exceeding any genome's capacity."""
    quotas = [0.0] * len(capacities)
    remaining, left = total, len(capacities)
    for i in sorted(range(len(capacities)), key=lambda i: capacities[i]):
        quotas[i] = min(capacities[i], remaining / left)
        remaining -= quotas[i]
        left -= 1
    return quotas


def _prepare_from_manifest(args: argparse.Namespace, settings, sequence_length: int, output_dir: Path) -> int:
    """
    Build train/val/test HDF5 files from a genome manifest (see
    scripts/download_diverse_genomes.py): 8 taxonomic classes, split by genome.

    The manifest holds one genome per species, so assigning whole genomes to a
    split is also a species-level split — no species shows up on both sides.
    Fragments are sampled uniformly along each genome instead of taken from its
    first bases, and a genome-level quota keeps one big genome from dominating
    its class.
    """
    import csv
    import json
    import time

    from metapathpredict.config.settings import (
        NCBI_GROUP_TO_TAXON,
        TAXON_CLASSES,
        superclass_index_map,
    )
    from metapathpredict.data import OneHotEncoder

    manifest_path = Path(args.manifest)
    root = manifest_path.parent
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    by_class: dict[str, list[dict]] = {name: [] for name in TAXON_CLASSES}
    for row in rows:
        taxon = NCBI_GROUP_TO_TAXON.get(row["group"])
        if taxon is not None:
            by_class[taxon].append(row)
    missing = [name for name, genomes in by_class.items() if len(genomes) < 3]
    if missing:
        logger.error(f"Need at least 3 genomes per class to split by genome; too few for: {missing}")
        return 1

    ratios = {"train": settings.data.train_ratio, "val": settings.data.val_ratio, "test": settings.data.test_ratio}
    split_names = tuple(ratios)
    per_class = args.fragments_per_class
    rng = np.random.RandomState(42)
    encoder = OneHotEncoder()
    attrs = {
        "num_classes": len(TAXON_CLASSES),
        "class_names": json.dumps(TAXON_CLASSES),
        "sequence_length": sequence_length,
    }
    writers = {
        s: _SplitWriter(output_dir / f"encoded_{s}_{sequence_length}.hdf5", sequence_length, args.chunk_size, attrs)
        for s in split_names
    }
    test_fasta = open(output_dir / "test_fragments.fasta", "w")
    assignments: list[dict] = []
    counts = {name: {s: 0 for s in split_names} for name in TAXON_CLASSES}
    t0 = time.time()
    done = 0

    for label, name in enumerate(TAXON_CLASSES):
        genomes = sorted(by_class[name], key=lambda r: r["accession"])
        genome_split = _assign_sequence_splits(
            len(genomes), ratios["train"], ratios["val"], ratios["test"], rng
        )
        for split in split_names:
            members = [g for g, gs in zip(genomes, genome_split) if gs == split]
            budget = round(per_class * ratios[split])
            caps = [float(g["genome_size"]) // sequence_length for g in members]
            quotas = _allocate_quotas(caps, budget)
            if sum(caps) < budget:
                logger.warning(f"  {name}/{split}: only {sum(caps):.0f} windows exist for a budget of {budget}")

            for genome, quota in zip(members, quotas):
                size = float(genome["genome_size"])
                taken = 0
                for seq in _stream_fasta(root / genome["path"]):
                    n_windows = len(seq) // sequence_length
                    if n_windows == 0:
                        continue
                    expected = quota * len(seq) / size
                    k = min(n_windows, int(expected) + int(rng.random_sample() < expected - int(expected)))
                    for w in rng.choice(n_windows, size=k, replace=False) if k else []:
                        frag = seq[w * sequence_length : (w + 1) * sequence_length].upper()
                        valid = sum(frag.count(c) for c in "ACGT")
                        if valid < 0.9 * sequence_length:
                            continue
                        writers[split].append(encoder.encode(frag).T, label)
                        if split == "test":
                            test_fasta.write(f">{name}_{counts[name][split]}|label={label}|acc={genome['accession']}\n{frag}\n")
                        counts[name][split] += 1
                        taken += 1
                assignments.append({
                    "accession": genome["accession"], "class": name, "split": split,
                    "species_taxid": genome["species_taxid"], "organism": genome["organism"],
                    "fragments": taken,
                })
                done += 1
                if done % 25 == 0:
                    logger.info(f"  {done}/{len(rows)} genomes, {time.time() - t0:.0f}s")
        logger.info(f"{name}: " + ", ".join(f"{s}={counts[name][s]}" for s in split_names))

    for w in writers.values():
        w.close()
    test_fasta.close()

    # No species may appear in more than one split.
    species_splits: dict[str, set] = {}
    for a in assignments:
        species_splits.setdefault(a["species_taxid"], set()).add(a["split"])
    shared = [sp for sp, sp_splits in species_splits.items() if len(sp_splits) > 1]
    if shared:
        logger.error(f"Species present in more than one split: {shared[:10]}")
        return 1
    logger.info(f"Verified: {len(species_splits)} species, none shared across train/val/test")

    with open(output_dir / "split_assignments.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(assignments[0]), delimiter="\t")
        w.writeheader()
        w.writerows(assignments)

    split_totals = {s: sum(counts[n][s] for n in TAXON_CLASSES) for s in split_names}
    metadata = {
        "sequence_length": sequence_length,
        "num_classes": len(TAXON_CLASSES),
        "class_names": TAXON_CLASSES,
        "superclass_of_class": superclass_index_map(TAXON_CLASSES),
        "train_size": split_totals["train"],
        "val_size": split_totals["val"],
        "test_size": split_totals["test"],
        "fragments": counts,
        "genomes_per_class_split": {
            n: {s: sum(1 for a in assignments if a["class"] == n and a["split"] == s) for s in split_names}
            for n in TAXON_CLASSES
        },
        "split_method": "genome-level, one genome per species (species-disjoint)",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Dataset prepared in {output_dir}: {split_totals} ({time.time() - t0:.0f}s)")
    return 0


def prepare_command(args: argparse.Namespace) -> int:
    """Prepare dataset from FASTA files (streaming, supports millions of fragments).

    Splits are assigned per source sequence (genome/chromosome/contig), not per
    fragment — see _assign_sequence_splits.
    """
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
    train_ratio = settings.data.train_ratio
    val_ratio = settings.data.val_ratio
    test_ratio = settings.data.test_ratio

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        return _prepare_from_manifest(args, settings, sequence_length, output_dir)
    if not args.inputs:
        logger.error("Provide FASTA inputs or --manifest")
        return 1

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
    rng = np.random.RandomState(42)
    split_names = ("train", "val", "test")
    split_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    # Open growable HDF5 outputs for all three splits up front — fragments are
    # written straight to their assigned split, no temp-file shuffle pass needed.
    out_paths = {s: output_dir / f"encoded_{s}_{sequence_length}.hdf5" for s in split_names}
    files = {s: h5py.File(out_paths[s], "w") for s in split_names}
    datasets: dict[str, dict] = {}
    for s in split_names:
        datasets[s] = {
            "seq": files[s].create_dataset(
                "sequences", shape=(0, 4, sequence_length), maxshape=(None, 4, sequence_length),
                dtype="float32", chunks=(min(chunk_size, 1024), 4, sequence_length),
                compression="gzip", compression_opts=1,
            ),
            "lab": files[s].create_dataset(
                "labels", shape=(0,), maxshape=(None,),
                dtype="int64", chunks=(min(chunk_size, 4096),),
                compression="gzip", compression_opts=1,
            ),
            "written": 0,
            "buf_data": [],
            "buf_labels": [],
        }
        # HDF5SequenceDataset.num_classes reads these attrs (falling back to 3
        # when absent); previously nothing ever set them.
        files[s].attrs["num_classes"] = 3
        files[s].attrs["sequence_length"] = sequence_length

    def flush(split: str) -> None:
        d = datasets[split]
        if not d["buf_data"]:
            return
        chunk_arr = np.stack(d["buf_data"], axis=0).astype(np.float32)
        chunk_lab = np.array(d["buf_labels"], dtype=np.int64)
        n = len(chunk_arr)
        d["seq"].resize(d["written"] + n, axis=0)
        d["lab"].resize(d["written"] + n, axis=0)
        d["seq"][d["written"] : d["written"] + n] = chunk_arr
        d["lab"][d["written"] : d["written"] + n] = chunk_lab
        d["written"] += n
        d["buf_data"].clear()
        d["buf_labels"].clear()

    def append(split: str, fragment_arr: np.ndarray, label: int) -> None:
        d = datasets[split]
        d["buf_data"].append(fragment_arr)
        d["buf_labels"].append(label)
        if len(d["buf_data"]) >= chunk_size:
            flush(split)

    class_counts = {0: 0, 1: 0, 2: 0}
    class_split_counts = {c: {s: 0 for s in split_names} for c in (0, 1, 2)}

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

        n_sequences = sum(1 for line in open(fasta_path) if line.startswith(">"))
        seq_split = _assign_sequence_splits(n_sequences, train_ratio, val_ratio, test_ratio, rng)
        if n_sequences <= 2:
            logger.warning(
                f"  Only {n_sequences} sequence(s) in {fasta_path.name} — "
                f"can't give every split real genome diversity from this file"
            )

        # Per-class fragment budget is split across train/val/test by the same
        # ratios, so a handful of early train-assigned sequences hitting
        # max_per_class can't starve val/test of their share.
        split_caps = {
            s: (round(max_per_class * split_ratios[s]) if max_per_class else None)
            for s in split_names
        }

        seq_count = 0
        for seq in _stream_fasta(fasta_path):
            split = seq_split[seq_count]
            seq_count += 1
            cap = split_caps[split]
            if cap is not None and class_split_counts[class_label][split] >= cap:
                continue
            try:
                cleaned, stats = preprocessor.process(seq)
                if cleaned is None:
                    continue
                for fragment, _start, _end in preprocessor.fragment(
                    cleaned, sequence_length, step_size=sequence_length
                ):
                    encoded = encoder.encode(fragment)  # (seq_len, 4)
                    append(split, encoded.T, class_label)  # transpose to (4, seq_len) for Conv1d
                    class_split_counts[class_label][split] += 1

                    if cap is not None and class_split_counts[class_label][split] >= cap:
                        break

            except Exception as e:
                logger.warning(f"  Failed to encode sequence: {e}")

            if max_per_class and all(
                split_caps[s] is not None and class_split_counts[class_label][s] >= split_caps[s]
                for s in split_names
            ):
                break

        class_counts[class_label] = sum(class_split_counts[class_label].values())
        logger.info(f"  Processed {seq_count} sequences → {class_counts[class_label]} fragments")

    for s in split_names:
        flush(s)

    total_written = sum(class_counts.values())
    if total_written == 0:
        logger.error("No sequences processed")
        for f in files.values():
            f.close()
        for p in out_paths.values():
            p.unlink(missing_ok=True)
        return 1

    for f in files.values():
        f.close()

    logger.info(f"Total: {total_written} fragments written")
    canonical_names = {0: "bacteria", 1: "eukaryotic", 2: "virus"}
    for label, name in canonical_names.items():
        if class_counts[label] > 0:
            logger.info(f"  {name}: {class_counts[label]}")

    split_totals = {s: sum(class_split_counts[c][s] for c in (0, 1, 2)) for s in split_names}
    for s in split_names:
        logger.info(f"  {s}: {split_totals[s]} samples → {out_paths[s]}")

    # Save metadata
    metadata = {
        "sequence_length": sequence_length,
        "num_classes": 3,
        "class_names": ["bacteria", "eukaryotic", "virus"],
        "total_fragments": total_written,
        "train_size": split_totals["train"],
        "val_size": split_totals["val"],
        "test_size": split_totals["test"],
        "class_counts": {canonical_names[k]: v for k, v in class_counts.items()},
        "split_method": "genome-level (per source sequence, not per fragment)",
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset prepared in {output_dir}")
    logger.info(f"  Train: {split_totals['train']}, Val: {split_totals['val']}, Test: {split_totals['test']}")

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

    class_names = list(getattr(model, "class_names", test_dataset.class_names))
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

    label_ids = list(range(len(class_names)))
    report_kwargs = dict(labels=label_ids, target_names=class_names, zero_division=0)

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, **report_kwargs))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds, labels=label_ids)
    print(cm)

    # Roll fine-grained classes up to prokaryote/eukaryote/virus — the level
    # other tools (DeepMicroClass, Tiara) report at, so results are comparable.
    from metapathpredict.config.settings import SUPERCLASSES, superclass_index_map

    super_results = None
    super_map = superclass_index_map(class_names)
    if super_map is not None and len(class_names) != len(SUPERCLASSES):
        to_super = np.array(super_map)
        super_targets, super_preds = to_super[all_targets], to_super[all_preds]
        super_ids = list(range(len(SUPERCLASSES)))
        super_kwargs = dict(labels=super_ids, target_names=SUPERCLASSES, zero_division=0)
        print(f"\nAggregated to {SUPERCLASSES}: accuracy {(super_targets == super_preds).mean():.4f}")
        print(classification_report(super_targets, super_preds, **super_kwargs))
        print(confusion_matrix(super_targets, super_preds, labels=super_ids))
        super_results = {
            "accuracy": float((super_targets == super_preds).mean()),
            "classification_report": classification_report(
                super_targets, super_preds, output_dict=True, **super_kwargs
            ),
        }

    if args.output:
        output_path = Path(args.output)
        eval_results = {
            "accuracy": float(accuracy),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                all_targets, all_preds, output_dict=True, **report_kwargs
            ),
            "aggregated_3class": super_results,
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
    train_parser.add_argument("--max-threads", type=int, default=16,
                              help="Cap on torch CPU threads and DataLoader workers")
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
    prepare_parser.add_argument("inputs", nargs="*", help="Input FASTA files (omit when using --manifest)")
    prepare_parser.add_argument("--manifest", help="Genome manifest TSV from scripts/download_diverse_genomes.py "
                                "(8 taxonomic classes, split by genome/species)")
    prepare_parser.add_argument("--fragments-per-class", type=int, default=30000,
                                help="With --manifest: target fragments per class across all splits")
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
