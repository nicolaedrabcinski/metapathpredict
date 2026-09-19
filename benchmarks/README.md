# Benchmarks

## Superseded: `our_model_eval*.json`, `deepmicroclass_predictions.tsv`, `tiara_*.txt`

These were produced on a "held-out" set that turned out to be contaminated: the script that built
it tried to exclude already-used genomes by comparing sequence IDs (`NC_...`) against assembly IDs
(`GCF_...`), which never match, so nothing was excluded. Checking afterwards, 89 of 94 eukaryotic,
10 of 10 viral and 1 of 14 bacterial held-out sequences were the same sequences the model had
trained on. The "our model 90% vs DeepMicroClass 77.6%" comparison therefore says nothing about
generalization, and the eukaryotic class in that data was fungi only.

They are kept only so the mistake stays visible. Do not cite them.

## Current protocol

`scripts/download_diverse_genomes.py` downloads one genome per species (at most two per genus) for
eight NCBI groups. `metapathpredict prepare --manifest` assigns whole genomes to train/val/test, so
the val and test splits are species-disjoint from training, and verifies that no species appears in
two splits. The test split is also written as `test_fragments.fasta` for running other tools.

## Current results: `taxa8_*_test.json`

Species-disjoint test split of `data/datasets/taxa8` (35,268 fragments of 500bp from 235 genomes,
8 classes, chance = 12.5%). Accuracy is on 8 classes, then rolled up to prokaryote/eukaryote/virus.

| Run | Checkpoint | 8-class | 3-class roll-up | Virus recall |
|---|---|---|---|---|
| 15 epochs per phase (still improving at the cap) | contrastive + probe | 59.0% | 78.1% | 59% |
| | RL | 53.7% | 76.5% | 4% |
| 50-epoch ceiling, patience 7 (contrastive stopped at 47, RL at 35), 15 probe epochs, 64k RL episodes per epoch | contrastive + probe | 60.7% | **82.5%** | 44% |
| (`taxa8_50ep_*.json`) | RL | **63.2%** | 80.4% | 56% |

Protozoa is the weakest class (recall 22-27% at best). Per-class genome counts in val/test are small
(5-6 for plant and vertebrate), so per-class numbers carry real noise.
