"""
Reference points for the contrastive pipeline: k-mer composition and a plain cross-entropy CNN.

Without them a contrastive result has nothing to be compared to. On the species-disjoint taxa8 split a
gradient-boosted 4-mer model already reaches about the accuracy of the contrastive encoder, so the
deep model has to be judged against it.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metapathpredict.models.contrastive import ContrastiveAugmentation, reverse_complement

logger = logging.getLogger(__name__)


def kmer_frequencies(sequences: np.ndarray, k: int = 4, chunk: int = 20000) -> np.ndarray:
    """
    Normalised k-mer counts of one-hot sequences [n, 4, length] (channels A, C, G, T).

    Windows that contain an all-zero column (N) are skipped. Returns float32 [n, 4**k]; a row sums
    to 1 (or 0 if the sequence has no complete window).
    """
    sequences = np.asarray(sequences)
    n, _, length = sequences.shape
    windows = length - k + 1
    out = np.zeros((n, 4 ** k), dtype=np.float32)
    for start in range(0, n, chunk):
        block = sequences[start:start + chunk]
        m = block.shape[0]
        base = block.argmax(axis=1).astype(np.int64)
        base[block.sum(axis=1) == 0] = -1
        code = np.zeros((m, windows), dtype=np.int64)
        invalid = np.zeros((m, windows), dtype=bool)
        for j in range(k):
            column = base[:, j:j + windows]
            invalid |= column < 0
            code = code * 4 + np.where(column < 0, 0, column)
        flat = (code + np.arange(m)[:, None] * 4 ** k)[~invalid]
        counts = np.bincount(flat, minlength=m * 4 ** k).reshape(m, 4 ** k).astype(np.float32)
        out[start:start + m] = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return out


def load_backbone_weights(model: nn.Module, checkpoint_path) -> int:
    """
    Load the backbone of a contrastive checkpoint into a ConfigurableCNN classifier, leaving its
    classifier head freshly initialised. Returns the number of tensors loaded; raises if the
    architectures do not match (a silent partial load would make the comparison meaningless).
    """
    model = getattr(model, "base", model)  # a strand-sharing wrapper holds the CNN as .base
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = "encoder."
    state = {k[len(prefix):]: v for k, v in checkpoint["encoder_state_dict"].items()
             if k.startswith(prefix) and not k.startswith(prefix + "classifier.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("classifier.")]
    if missing or unexpected:
        raise ValueError(f"backbone mismatch: missing {missing[:3]}, unexpected {unexpected[:3]}")
    return len(state)


def _augment(x: torch.Tensor, mode: str, augmentation: ContrastiveAugmentation | None) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "rc":
        flip = torch.rand(x.size(0), device=x.device) < 0.5
        return torch.where(flip[:, None, None], reverse_complement(x), x)
    if mode == "full":
        return augmentation(x)[0]
    raise ValueError(f"augment must be none, rc or full, got {mode!r}")


def evaluate_accuracy(model: nn.Module, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Accuracy, targets and predictions of `model` (eval mode) over a loader."""
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch[0].to(device)).argmax(dim=1).cpu())
            targets.append(batch[1])
    targets, preds = torch.cat(targets).numpy(), torch.cat(preds).numpy()
    return float((targets == preds).mean()), targets, preds


class WithAuxTargets(torch.utils.data.Dataset):
    """A dataset of (x, y) pairs that also returns an auxiliary label per item: (x, y, aux)."""

    def __init__(self, base, aux: np.ndarray):
        if len(base) != len(aux):
            raise ValueError(f"{len(aux)} auxiliary labels for {len(base)} items")
        self.base, self.aux = base, torch.as_tensor(aux, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i][:2]
        return x, y, self.aux[i]


def predict_probabilities(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Targets [n] and softmax probabilities [n, classes] of `model` (eval mode) over a loader."""
    model.eval()
    targets, probs = [], []
    with torch.no_grad():
        for batch in loader:
            probs.append(F.softmax(model(batch[0].to(device)), dim=1).cpu())
            targets.append(batch[1])
    return torch.cat(targets).numpy(), torch.cat(probs).numpy()


def save_supervised_checkpoint(model: nn.Module, path, config: dict) -> None:
    """Weights of a ConfigurableCNN classifier with what is needed to rebuild it (see load_supervised_checkpoint)."""
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def build_classifier(num_classes: int, backbone: str = "large", base_channels: int = 128, norm: str = "batch",
                     pool: str = "avg", rc_share: str = "none") -> nn.Module:
    """A ConfigurableCNN classifier, optionally looking at both strands with shared weights (`rc_share`: mean | max)."""
    from metapathpredict.models.configurable_cnn import ConfigurableCNN, RCShared

    model = ConfigurableCNN(in_channels=4, num_classes=num_classes, kernel_preset=backbone,
                            base_channels=base_channels, norm=norm, pool=pool)
    return model if rc_share == "none" else RCShared(model, rc_share)


def load_supervised_checkpoint(path, device="cpu") -> nn.Module:
    """Rebuild the classifier saved by save_supervised_checkpoint, in eval mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_classifier(config["num_classes"], config["backbone"], config["base_channels"], config.get("norm", "batch"),
                             config.get("pool", "avg"), config.get("rc_share", "none"))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def train_supervised(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    epochs: int = 20,
    patience: int = 7,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    augment: str = "rc",
    augmentation: ContrastiveAugmentation | None = None,
    sink=None,
    lr_schedule: str = "constant",
    aux_head: nn.Module | None = None,
    aux_weight: float = 0.3,
) -> dict:
    """
    Train `model` (a classifier: [batch, 4, length] -> logits) with cross-entropy and keep the weights
    of the epoch with the best validation accuracy. `augment`: none | rc (random reverse complement) |
    full (the contrastive augmentation, first view only). `lr_schedule`: constant | cosine (annealed to 0 over
    `epochs`). With `aux_head` (a module mapping the pooled embedding to auxiliary classes) and batches of
    (x, y, aux), the loss adds `aux_weight` times the cross-entropy of the auxiliary labels (-100 is ignored):
    a second, finer supervision signal that shapes the shared features and is thrown away afterwards.
    Returns {"best_val_acc", "best_epoch", "history"}.
    """
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"lr_schedule must be constant or cosine, got {lr_schedule!r}")
    if augment == "full" and augmentation is None:
        augmentation = ContrastiveAugmentation()
    model.to(device)
    if aux_head is not None:
        aux_head.to(device)
    parameters = list(model.parameters()) + (list(aux_head.parameters()) if aux_head is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if lr_schedule == "cosine" else None
    best_acc, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []

    for epoch in range(1, epochs + 1):
        model.train()
        if aux_head is not None:
            aux_head.train()
        total, correct, seen = 0.0, 0, 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            x = _augment(x, augment, augmentation)
            if aux_head is not None and len(batch) > 2:
                embedding = model.get_embeddings(x)
                logits = model.classifier(embedding)
                loss = F.cross_entropy(logits, y)
                aux = batch[2].to(device)
                if (aux != -100).any():
                    loss = loss + aux_weight * F.cross_entropy(aux_head(embedding), aux, ignore_index=-100)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += len(y)

        if scheduler is not None:
            scheduler.step()
        val_acc, _, _ = evaluate_accuracy(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total / seen, "train_acc": correct / seen, "val_acc": val_acc}
        history.append(row)
        if sink is not None:
            sink.log_metrics({"supervised/train_loss": row["train_loss"], "supervised/train_acc": row["train_acc"],
                              "supervised/val_acc": val_acc}, step=epoch)
        logger.info(f"[Supervised] Epoch {epoch}/{epochs} | loss {row['train_loss']:.4f} | "
                    f"train acc {row['train_acc']:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if patience and stale >= patience:
                logger.info(f"Early stopping: val accuracy has not improved for {patience} epochs")
                break

    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_acc": best_acc, "best_epoch": best_epoch, "history": history}
