# Plain CNN (Cross-Entropy)

A `ConfigurableCNN` trained end-to-end with cross-entropy on the 8 taxonomic classes (bacteria,
archaea, fungi, protozoa, plant, invertebrate, vertebrate, virus). No contrastive pretraining, no
RL. In our own measurements this has been the strongest of the three approaches in this repository
— see `benchmarks/README.md` for the current numbers before assuming a fancier approach wins.

## Usage

The supported entry point is `scripts/baselines.py`, not `metapathpredict train` (that command only
exposes the contrastive/RL pipelines, see [overview](overview.md)):

```bash
python scripts/baselines.py supervised \
    --data-dir data/datasets/taxa8fam500_s1 \
    --augment rc \
    --rc-share mean \
    --save-checkpoint
```

This is `metapathpredict.baselines.build_classifier` + `train_supervised`, called for you with
genome-level evaluation on the test split at the end. Programmatically:

```python
from metapathpredict.baselines import build_classifier, train_supervised

model = build_classifier(
    num_classes=8,
    backbone="large",       # "small" | "medium" | "large" | "progressive" | "multi"
    base_channels=128,
    norm="batch",           # "batch" | "group"
    pool="avg",             # "avg" | "max" | "avgmax"
    rc_share="mean",        # "none" | "mean" | "max" — shared weights for a sequence
)                           # and its reverse complement; the one change that has reliably helped

result = train_supervised(
    model, train_loader, val_loader, device="cuda",
    epochs=20, patience=7, lr=1e-3, augment="rc",   # "none" | "rc" | "full"
)
```

`build_classifier` wraps `metapathpredict.models.configurable_cnn.ConfigurableCNN`
(`kernel_preset` picks the receptive field: `small`/`medium`/`large`, or `progressive`/`multi` for
multiple kernel sizes) in `RCShared` when `rc_share` is not `"none"`. `augment="rc"` randomly
replaces a fraction of training sequences with their reverse complement each epoch; `"full"` uses
the same crop/mutate/mask augmentation as the contrastive phase (`ContrastiveAugmentation`) and has
not helped here — see `benchmarks/README.md`.

## Reference points

`scripts/baselines.py kmer` trains a k-mer-composition baseline (logistic regression or gradient
boosting on 4-mer or 5-mer frequencies) for comparison — the plain CNN needs to clear this bar by a
real, statistically significant margin to be worth its cost; see `scripts/compare_predictions.py`
for the paired-bootstrap test that checks this.

## What did not help here

Tried the same way, on the same splits (full log in `BACKLOG.md`): `pool="max"` or `"avgmax"`
instead of `"avg"`; wider backbones (256 base channels) or the `multi`/`progressive` kernel presets;
an auxiliary taxonomic-lineage classification head; a stacked meta-classifier over several trained
models (plain probability averaging won instead). What did help: `rc_share="mean"`, an ensemble of
several training seeds, and — by far the largest single effect — more diverse training genomes for
the weakest class (virus).
