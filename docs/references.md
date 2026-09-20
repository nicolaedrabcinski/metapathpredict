# References

Papers behind design decisions in the contrastive phase, taken from the reading list
[Contrastive-Learning-NLP-Papers](https://github.com/ryanzhumich/Contrastive-Learning-NLP-Papers)
(foundation section; the NLP-specific part of that list is not relevant here). All results below are
from the papers, on images or text; whether they carry over to DNA fragments is tested in this repo,
not assumed.

| Paper | What it says | Where it matters here |
|---|---|---|
| [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) (Khosla et al.) | Summing over positives outside the log (L_out) gets 78.7% vs 67.4% for the variant with the sum inside (ImageNet, batch 6144). Larger batches help; 2048 is enough for most purposes. | `SupConLoss` is the L_out form. |
| [Perfectly Balanced](https://arxiv.org/abs/2204.07596) (Chen et al.) | SupCon suffers class collapse: within-class structure is lost, which hurts transfer from coarse to fine labels. | Our labels are taxon groups; the real structure is species. |
| [Debiased Contrastive Learning](https://arxiv.org/abs/2007.00224) (Chuang et al.) | Randomly drawn negatives include same-class pairs; corrects the loss with the class prior tau+. | `contrastive.debias_tau_plus` (1/8 for our 8 balanced classes). |
| [Hard Negative Samples](https://arxiv.org/abs/2010.04592) (Robinson et al.) | Reweights negatives by exp(beta * similarity); a few lines of code, no extra compute. beta=0 recovers the debiased loss. They use beta 0.5-1 with tau+ 0.05-0.1. | `contrastive.hard_negative_beta`. |
| [Alignment and Uniformity](https://arxiv.org/abs/2005.10242) (Wang, Isola) | Two metrics on the hypersphere that track downstream quality. | Logged every epoch; comparable across batch sizes, unlike the NT-Xent value. |
| [Dimensional Collapse](https://arxiv.org/abs/2110.09348) (Jing et al.) | Embeddings can span a low-dimensional subspace even with negatives; caused by strong augmentation and implicit regularization. Diagnose with the singular value spectrum. | Effective rank is logged; the embedding std alone does not show this. |
| [What Makes for Good Views](https://arxiv.org/abs/2005.10243) (Tian et al.) | Views should share as little information as possible while keeping what the task needs; there is a sweet spot. | Augmentation strength is a hyperparameter, not a constant. |
| [SimCLR](https://arxiv.org/abs/2002.05709), [MoCo](https://arxiv.org/abs/1911.05722) | Large batch or a queue of negatives (K=65536, momentum 0.999) plus a non-linear projection head and composed augmentations. | Batch-size sweep; a queue is the alternative when the GPU is already compute-bound. |
| [Intriguing Properties](https://proceedings.neurips.cc/paper/2021/file/628f16b29939d1b060af49f66ae0f7f8-Paper.pdf) (Chen et al.) | An easy feature shared across views can suppress learning of others. | Hypothesis, untested: GC content survives reverse-complement and mutation. |
| [Rethinking InfoNCE](https://arxiv.org/abs/2105.13003) (Wu et al.) | With noisy negatives, more negatives is not always better. | Why the batch-size sweep may not be monotonic. |

## Reinforcement learning

- [Spinning Up: Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)
- [Algorithms of Reinforcement Learning](https://sites.ualberta.ca/~szepesva/rlbook.html) (Szepesvari)

The RL phase here is a contextual bandit (one fragment, one answer, one reward), so most of that list
(DQN, TRPO/PPO, exploration, hierarchy, memory, model-based) does not apply. What does: variance and
seeds ("Deep RL that Matters"), sampling hard examples more often (prioritized replay), and reward
shaping as the one thing RL can do that cross-entropy cannot. In practice a plain cross-entropy CNN
has consistently matched or beaten the contrastive+RL pipeline on this task — see `benchmarks/README.md`.

## Architecture and taxonomic classification (added while investigating why RC-share and virus
## diversity helped; abstracts only, not read in full — see `BACKLOG.md` for what was checked)

| Paper | What it says | Where it matters here |
|---|---|---|
| [VirHunter](https://www.frontiersin.org/articles/10.3389/fbinf.2022.867111/full) (Sukhorukov et al. 2022) | Three CNNs (k=5,7,10) + a shallow random forest stacker; global max-pooling; bacteriophage fragments get misclassified as bacteria in their leave-one-family-out setting too. | Motivated `scripts/stacking_eval.py` (the stacker did not beat plain averaging here) and the pooling sweep (max/avg+max did not beat mean pooling here either). |
| [Caduceus](https://arxiv.org/abs/2403.03234) (Schiff et al.) | A DNA state-space model built to be reverse-complement equivariant by construction. | Matches our `RCShare` finding (shared weights for a sequence and its reverse complement): +1.8pp, the one architecture change this session that reliably helped. |
| [ConvNova](https://arxiv.org/abs/2502.18538) | Argues CNNs remain competitive with transformers/SSMs on DNA foundation-model benchmarks when using dilated and gated convolutions; our backbone uses neither. | Candidate next architecture change — `ConvBlock` already accepts a `dilation` parameter, unused by `ConfigurableCNN`. |
| [TaxDistill](https://arxiv.org/abs/2605.28868) | Distilling soft labels from a 500M-parameter genomic foundation model (GenomeOcean) into a small student raised F1 from 0.763 to 0.941 on CAMI2 metagenomic taxonomy. | Not tried; would need access to a pretrained genomic foundation model. |
| [Metaxa](https://www.biorxiv.org/content/10.64898/2026.04.20.719780v1.full) (2026, long-read Nanopore) | A transformer classifier generalizes to out-of-sample (novel) organisms better than Kraken2/MetageNN at genus level. | Consistent with our own finding (`scripts/novelty_distance.py`): accuracy falls off sharply with sequence distance to the nearest training genome. |
