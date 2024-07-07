import concurrent.futures
import random
import sys
import json

from pathlib import Path
from Bio import SeqIO
from sklearn.utils import shuffle
from utils import preprocess as pp

def prepare_dataset(label, in_seqs, out_path, fragment_length, n_frags, random_seed, n_cpus):
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=in_seqs,
        fragment_length=fragment_length,
        n_frags=n_frags,
        label=label,
        label_int=1 if label == 'virus' else 0 if label == 'eucaryotic' else 2,
        random_seed=random_seed,
        n_cpus=n_cpus
    )
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    out_file = Path(out_path, f"seqs_{label}_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_file, "fasta")

def prepare_datasets(config):
    with open(config, "r") as jsonfile:
        cf = json.load(jsonfile)

    datasets = [
        ("virus", cf["prepare_ds_virus"]["path_virus"], cf["prepare_ds_virus"]["virus_out_path"]),
        ("eucaryotic", cf["prepare_ds_eucaryotic"]["path_eucaryotic"], cf["prepare_ds_eucaryotic"]["eucaryotic_out_path"]),
        ("bacteria", cf["prepare_ds_bacteria"]["path_bact"], cf["prepare_ds_bacteria"]["bacteria_out_path"]),
    ]
    fragment_lengths = [500, 1000]
    n_frags = 20000
    random_seed = cf["prepare_ds_virus"]["random_seed"]
    n_cpus = cf["prepare_ds_virus"]["n_cpus"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_cpus) as executor:
        futures = []
        for label, in_seqs, out_path in datasets:
            for fragment_length in fragment_lengths:
                futures.append(executor.submit(prepare_dataset, label, in_seqs, out_path, fragment_length, n_frags, random_seed, n_cpus))

        for future in concurrent.futures.as_completed(futures):
            future.result()

    print("Dataset preparation completed.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    prepare_datasets(config_file)
