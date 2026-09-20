# Training Approaches Overview

MetaPathPredict trains a classifier for 500 bp DNA fragments into 8 taxonomic classes (bacteria,
archaea, fungi, protozoa, plant, invertebrate, vertebrate, virus; rolled up to
prokaryote/eukaryote/virus). Two approaches are implemented and actively used; a third
(`Configurable CNN` trained directly with cross-entropy, no contrastive step) turned out to be the
strongest of the three in our own measurements — see the numbers below before picking one.

## Comparison

| Approach | What it is | Labeled data | Current standing |
|----------|------------|---------------|-------------------|
| **Plain CNN** (`scripts/baselines.py supervised`) | `ConfigurableCNN` trained end-to-end with cross-entropy | Required | Currently our best recipe; see `benchmarks/README.md` |
| **Contrastive pretraining** (`metapathpredict train --pipeline contrastive`) | Self-supervised NT-Xent/SupCon pretraining of `ContrastiveEncoder`, then a linear probe | Only for the probe | Has not beaten the plain CNN on this task so far |
| **RL fine-tuning** (`metapathpredict train --pipeline rl`) | An actor-critic agent fine-tuned from a contrastive encoder; the task is a one-step contextual bandit (one fragment, one action, one reward), not a multi-step RL problem | Required | Has not beaten the contrastive probe or the plain CNN in any run so far |

## 1. Plain CNN (cross-entropy)

```bash
python scripts/baselines.py supervised --augment rc --rc-share mean --save-checkpoint
```

`ConfigurableCNN` (kernel presets `small`/`medium`/`large`/`progressive`/`multi`), trained directly
with cross-entropy. `--rc-share mean` shares weights between a sequence and its reverse complement
(the one architecture change that has reliably helped, see `benchmarks/README.md`).

[Learn more →](cnn.md)

## 2. Contrastive Learning

```bash
metapathpredict train --pipeline contrastive --config configs/train_gpu.yaml
```

`ContrastiveEncoder` + `NTXentLoss`/`SupConLoss` (`metapathpredict.models.contrastive`), pretrained
without labels, then a linear probe (`metapathpredict.probe`) fit on the frozen encoder. Useful when
labeled data is scarce relative to unlabeled sequence; on our current dataset size it has not
outperformed training the same CNN directly on the labels.

[Learn more →](contrastive.md)

## 3. Deep Reinforcement Learning

```bash
metapathpredict train --pipeline rl --config configs/train_gpu.yaml
```

DQN, Policy Gradient or Actor-Critic (`metapathpredict.models.reinforcement`), fine-tuned from a
contrastive encoder checkpoint. Because the task has no real multi-step structure (one fragment, one
decision, no dependence on earlier decisions), most of what distinguishes RL from a classifier —
exploration, credit assignment across steps, planning — has nothing to act on here; treat it as an
experiment, not the default choice.

[Learn more →](reinforcement.md)

## Current numbers

See `benchmarks/README.md` for the actual, genome-level-bootstrapped results on the current
family-disjoint test split, and `BACKLOG.md` for the full log of what was tried and what did or did
not help.
