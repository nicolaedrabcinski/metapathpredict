# Benchmarks

## Superseded: `our_model_eval*.json`, `deepmicroclass_predictions.tsv`, `tiara_*.txt`

These were produced on a "held-out" set that turned out to be contaminated: the script that built
it tried to exclude already-used genomes by comparing sequence IDs (`NC_...`) against assembly IDs
(`GCF_...`), which never match, so nothing was excluded. Checking afterwards, 89 of 94 eukaryotic,
10 of 10 viral and 1 of 14 bacterial held-out sequences were the same sequences the model had
trained on. The "our model 90% vs DeepMicroClass 77.6%" comparison therefore says nothing about
generalization, and the eukaryotic class in that data was fungi only.

They are kept only so the mistake stays visible. Do not cite them.

## Also superseded: `taxa8_*_test.json`

These are a genuine species-disjoint split (`data/datasets/taxa8`, 35,268 fragments of 500bp from
235 genomes, 8 classes) — not contaminated — but weaker than the current protocol in two ways:
species-disjoint still lets a test genome share a genus or family with training, and the numbers
below are plain accuracy with no genome-level uncertainty. Kept for reference, not cited as current.

| Run | Checkpoint | 8-class | 3-class roll-up | Virus recall |
|---|---|---|---|---|
| 15 epochs per phase | contrastive + probe | 59.0% | 78.1% | 59% |
| | RL | 53.7% | 76.5% | 4% |
| 50-epoch ceiling | contrastive + probe | 60.7% | 82.5% | 44% |
| | RL | 63.2% | 80.4% | 56% |

## Current protocol

`scripts/download_diverse_genomes.py` downloads genomes per NCBI group (`--extend` adds more,
picked for taxonomic novelty). `metapathpredict prepare --manifest --split-by family` assigns whole
**families** (not just species) to train/val/test — a test genome then has no relative of the same
family in training — and verifies no species or family crosses a split. `--fixed-splits` adds
genomes to an existing dataset without moving genomes already assigned. `--split-seed` gives
independent splits (`taxa8fam500_s1/s2/s3`) to see how much a result depends on which genomes ended
up in the test set.

Evaluation is genome-level, not fragment-level: fragments of one genome are right or wrong
together, so a plain accuracy overstates certainty. `metapathpredict.genome_eval` (used by
`metapathpredict evaluate`, `scripts/baselines.py`, `scripts/run_experiment.py`,
`scripts/final_ensemble.py`) bootstraps over genomes for a 95% interval, and — when
`split_assignments.tsv` has lineage columns (written by `prepare`, or added by
`scripts/annotate_lineages.py`) — reports accuracy by how close the nearest training genome of the
class is ("near" = shares a genus or family, "far" = the rest). `scripts/novelty_distance.py` goes
further with actual sequence distance (Mash) instead of taxonomic rank.

Reference points: `scripts/baselines.py kmer` (k-mer composition + gradient boosting) and
`scripts/baselines.py supervised` (plain CNN, no contrastive pretraining). `scripts/compare_predictions.py`
does a paired bootstrap between any two saved prediction files. `scripts/stacking_eval.py` tests
whether a meta-classifier beats averaging several models.

## Current best: RC-share + expanded virus set + ensemble

Family-disjoint test split of `data/datasets/taxa8vir_fam500_s1` (8 classes, 3000 virus genomes vs.
500 in `taxa8fam500_*`), CNN with shared weights for a sequence and its reverse complement
(`--rc-share mean`), averaged over 3 training seeds:

| | 8-class | 3-class roll-up | Virus recall |
|---|---|---|---|
| single model, no rc-share, original (500) virus set | 61.3% | — | 18.9% |
| single model, rc-share + expanded (3000) virus set | 62.6% [58.7, 66.2] | 82.0% | 52.6% |
| **ensemble of 3 seeds, rc-share + expanded virus set** | **65.6% [62.0, 68.7]** | **84.1% [81.7, 86.5]** | **60.9% [57.9, 63.8]** |

The ensemble beats the single model of the same recipe by +3.0pp [+1.9, +4.0] (paired bootstrap over
genomes, P=1.00). Confirmed on two further splits, one seed each: `taxa8fam500_s2` 61.1%/82.5%,
`taxa8fam500_s3` 63.5%/81.5%.

What did **not** help, tried the same way (see `BACKLOG.md` for the full log): an auxiliary
taxonomic-lineage head, wider/alternate backbones (`medium`/`progressive`/`multi`, 64/256 channels),
splitting the heterogeneous `protozoa` class into 4 eukaryotic supergroups, max/avg+max pooling, a
stacked meta-classifier over several models (plain averaging won).

Protozoa remains the weakest class (recall 35-40%) and has not improved under any variant tried so
far. Per-class genome counts in some classes are small (5-14 for plant/vertebrate/protozoa/archaea),
so per-class numbers carry real uncertainty — see the bootstrap intervals, not just the point estimate.
