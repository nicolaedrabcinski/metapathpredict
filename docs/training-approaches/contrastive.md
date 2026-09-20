# Contrastive Learning

Self-supervised pretraining of a `ConfigurableCNN` backbone (`ContrastiveEncoder`), followed by a
linear probe fit on the frozen encoder (`metapathpredict.probe`). Useful when labeled data is scarce
relative to unlabeled sequence; on the current dataset size it has **not** outperformed training the
same CNN directly on the labels (`scripts/baselines.py supervised`) — see `benchmarks/README.md`
before choosing this over the plain CNN.

## Losses

```python
from metapathpredict.models.contrastive import (
    ContrastiveEncoder, NTXentLoss, SupConLoss, ContrastiveAugmentation, ContrastiveTrainer,
)

encoder = ContrastiveEncoder(
    backbone="large",       # kernel preset, same options as ConfigurableCNN
    base_channels=128,
    projection_dim=512,
    hidden_dim=1024,
    num_classes=8,          # only used by the classifier head fit afterwards as a probe
)

augmentation = ContrastiveAugmentation(mutation_rate=0.1, mask_rate=0.15, crop_ratio=(0.8, 1.0))

trainer = ContrastiveTrainer(
    encoder, optimizer=torch.optim.AdamW(encoder.parameters(), lr=1e-4),
    augmentation=augmentation, temperature=0.07, device="cuda",
    use_supervised=False,   # False: NTXentLoss (SimCLR-style, unsupervised)
                             # True:  SupConLoss (uses the batch's labels) or a mix (see below)
)
```

`NTXentLoss` also supports two extensions, both from `docs/references.md`: `tau_plus` (debiasing,
Chuang et al. 2020 — the estimated chance a random negative is actually same-class) and `beta`
(hard-negative reweighting, Robinson et al. 2021), plus `decoupled=True` (removes the positive pair
from the denominator, Yeh et al. 2022 — `loss_type: "dcl"` in the Hydra configs). `ContrastiveTrainer`
also accepts `supcon_weight` for a weighted mix of `SupConLoss` and `NTXentLoss` in the same step
(`loss_type: "hybrid"`) rather than pure supervised or pure unsupervised.

## Training

```bash
metapathpredict train --pipeline contrastive --config configs/train_gpu.yaml
```

reads the `contrastive:` block of the config (see `configs/train_gpu.yaml`/`train_cpu.yaml` for a
complete example — `backbone`, `base_channels`, `loss_type`, `temperature`, `mutation_rate`,
`mask_rate`, `num_epochs`, `learning_rate`, `batch_size`, `early_stopping_patience`, `probe_epochs`).
For sweeps, `scripts/run_experiment.py` runs the same pipeline under Hydra
(`experiment_name=... contrastive.temperature=0.05,0.1 -m`) and logs every run to MLflow.

`ContrastiveTrainer.train_epoch` logs positive/negative cosine similarity, gradient norm and
embedding std every epoch (std near 0 flags representation collapse). `validate_epoch` additionally
computes alignment and uniformity (Wang & Isola 2020) and the effective rank of the projection and
backbone embeddings — a healthy per-dimension std can still hide dimensional collapse that effective
rank catches.

## Linear probe

```python
from metapathpredict.probe import fit_linear_probe

result = fit_linear_probe(encoder.encoder, train_loader, val_loader, device="cuda", epochs=100)
# result["best_val_acc"], result["best_epoch"]
```

Recalibrates BatchNorm running statistics on clean (non-augmented) data first, then fits the
classifier head on cached, frozen embeddings — fitting the head with the backbone in train mode
(BatchNorm using batch statistics) turned out to give a systematically inflated validation accuracy
in earlier runs here; `probe_mode: frozen` in the config uses this fixed version. See
`scripts/refit_probe.py` to re-score an existing checkpoint this way without retraining.
