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
