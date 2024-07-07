import os
import sys
import json

from Bio import SeqIO

import random
import numpy as np

from utils import preprocess as pp
from sklearn.utils import shuffle

import h5py

from pathlib import Path

def prepare_ds_nn(
        path_virus,
        path_eucaryotic,
        path_bact,
        out_path,
        fragment_length=1000,
        n_cpus=1,
        random_seed=None,
):
    """
    This is a function for the example dataset preparation.
    If random seed is not specified it will be generated randomly.
    """

    if random_seed is None:
        random.seed(a=random_seed)
        random_seed = random.randrange(1000000)
    random.seed(a=random_seed)

    v_encoded, v_encoded_rc, v_labs, v_seqs, v_n_frags = pp.prepare_ds_fragmenting(
        in_seq=path_virus, label='virus', label_int=1, fragment_length=fragment_length,
        sl_wind_step=int(fragment_length / 2), max_gap=0.05, n_cpus=n_cpus)


    pl_encoded, pl_encoded_rc, pl_labs, _, pl_n_frags = pp.prepare_ds_sampling(
        in_seqs=path_eucaryotic, fragment_length=fragment_length, n_frags=v_n_frags, label='eucaryotic', label_int=0,
        random_seed=random.randrange(1000000))

    b_encoded, b_encoded_rc, b_labs, b_seqs, b_n_frags = pp.prepare_ds_sampling(
        in_seqs=path_bact, fragment_length=fragment_length,
        n_frags=v_n_frags, label='bacteria', label_int=2, random_seed=random.randrange(1000000))

    assert v_n_frags == b_n_frags
    assert v_n_frags == pl_n_frags

    all_encoded = np.concatenate([v_encoded, pl_encoded, b_encoded])
    all_encoded_rc = np.concatenate([v_encoded_rc, pl_encoded_rc, b_encoded_rc])
    all_labs = np.concatenate([v_labs, pl_labs, b_labs])

    # adding reverse complement
    all_encoded = np.concatenate((all_encoded, all_encoded_rc))
    all_encoded_rc = np.concatenate((all_encoded_rc, all_encoded))
    all_labs = np.concatenate((all_labs, all_labs))

    # saving one-hot encoded fragments
    pp.storing_encoded(all_encoded, all_encoded_rc, all_labs,
                       Path(out_path, f"encoded_train_{fragment_length}.hdf5"))

def prepare_ds_rf_viral(
        path_virus,
        path_eucaryotic,
        path_bact,
        virus_out_path,
        eucaryotic_out_path,
        bacteria_out_path,
        fragment_length=1000,
        n_cpus=1,
        random_seed=1,
):
    n_frags = 20000
    # sampling virus
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_virus, 
        fragment_length=fragment_length,
        n_frags=n_frags, 
        label='virus', 
        label_int=1, 
        random_seed=random_seed, 
        n_cpus=n_cpus
        )
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    assert len(seqs) == n_frags
    out_path_seqs = Path(virus_out_path, f"seqs_virus_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")
    # sampling eucaryotic
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_eucaryotic, fragment_length=fragment_length,
        n_frags=n_frags, label='eucaryotic', label_int=0, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    # force same number of fragments
    assert len(seqs) == n_frags
    out_path_seqs = Path(virus_out_path, f"seqs_eucaryotic_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")
    # sampling bacteria
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_bact, fragment_length=fragment_length,
        n_frags=n_frags, label='bacteria', label_int=2, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    assert len(seqs) == n_frags
    out_path_seqs = Path(virus_out_path, f"seqs_bacteria_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")

def prepare_ds_rf_eucaryotic(
        path_virus,
        path_eucaryotic,
        path_bact,
        virus_out_path,
        eucaryotic_out_path,
        bacteria_out_path,
        fragment_length=1000,
        n_cpus=1,
        random_seed=1,
    ):
    n_frags = 20000
    # sampling virus
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_virus, 
        fragment_length=fragment_length,
        n_frags=n_frags, 
        label='eucaryotic', 
        label_int=1, 
        random_seed=random_seed, 
        n_cpus=n_cpus
        )
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    assert len(seqs) == n_frags
    out_path_seqs = Path(eucaryotic_out_path, f"seqs_virus_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")
    # sampling eucaryotic
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_eucaryotic, fragment_length=fragment_length,
        n_frags=n_frags, label='eucaryotic', label_int=0, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    # force same number of fragments
    assert len(seqs) == n_frags
    out_path_seqs = Path(eucaryotic_out_path, f"seqs_eucaryotic_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")
    # sampling bacteria
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_bact, fragment_length=fragment_length,
        n_frags=n_frags, label='bacteria', label_int=2, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    assert len(seqs) == n_frags
    out_path_seqs = Path(eucaryotic_out_path, f"seqs_bacteria_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")

def prepare_ds_rf_bacteria(
        path_virus,
        path_eucaryotic,
        path_bact,
        virus_out_path,
        eucaryotic_out_path,
        bacteria_out_path,
        fragment_length=1000,
        n_cpus=1,
        random_seed=1,
):
    n_frags = 20000
    # sampling virus
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_virus, 
        fragment_length=fragment_length,
        n_frags=n_frags, 
        label='bacteria', 
        label_int=1, 
        random_seed=random_seed, 
        n_cpus=n_cpus
        )
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    assert len(seqs) == n_frags
    out_path_seqs = Path(bacteria_out_path, f"seqs_virus_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")
    # sampling eucaryotic
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_eucaryotic, fragment_length=fragment_length,
        n_frags=n_frags, label='eucaryotic', label_int=0, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    # force same number of fragments
    assert len(seqs) == n_frags
    out_path_seqs = Path(bacteria_out_path, f"seqs_eucaryotic_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")
    # sampling bacteria
    _, _, _, seqs, _ = pp.prepare_ds_sampling(
        in_seqs=path_bact, fragment_length=fragment_length,
        n_frags=n_frags, label='bacteria', label_int=2, random_seed=random_seed, n_cpus=n_cpus)
    seqs = shuffle(seqs, random_state=random_seed, n_samples=n_frags)
    assert len(seqs) == n_frags
    out_path_seqs = Path(bacteria_out_path, f"seqs_bacteria_sampled_{fragment_length}_{n_frags}.fasta")
    SeqIO.write(seqs, out_path_seqs, "fasta")

def prepare_ds_virus(config):
    with open(config, "r") as jsonfile:
        cf = json.load(jsonfile)

    assert Path(cf["prepare_ds_virus"]["path_virus"]).exists(), f'{cf["prepare_ds_virus"]["path_virus"]} does not exist'
    assert Path(cf["prepare_ds_virus"]["path_eucaryotic"]).exists(), f'{cf["prepare_ds_virus"]["path_eucaryotic"]} does not exist'
    assert Path(cf["prepare_ds_virus"]["path_bact"]).exists(), f'{cf["prepare_ds_virus"]["path_bact"]} does not exist'

    Path(cf["prepare_ds_virus"]["virus_out_path"]).mkdir(parents=True, exist_ok=True)

    for l_ in 500, 1000:
        prepare_ds_nn(
            path_virus=cf["prepare_ds_virus"]["path_virus"],
            path_eucaryotic=cf["prepare_ds_virus"]["path_eucaryotic"],
            path_bact=cf["prepare_ds_virus"]["path_bact"],
            out_path=cf["prepare_ds_virus"]["virus_out_path"],
            fragment_length=l_,
            n_cpus=cf["prepare_ds_virus"]["n_cpus"],
            random_seed=cf["prepare_ds_virus"]["random_seed"],
        )
        prepare_ds_rf_viral(
            path_virus=cf["prepare_ds_virus"]["path_virus"],
            path_eucaryotic=cf["prepare_ds_virus"]["path_eucaryotic"],
            path_bact=cf["prepare_ds_virus"]["path_bact"],
            virus_out_path=cf["prepare_ds_virus"]["virus_out_path"],
            eucaryotic_out_path=cf["prepare_ds_eucaryotic"]["eucaryotic_out_path"],
            bacteria_out_path=cf["prepare_ds_bacteria"]["bacteria_out_path"],
            fragment_length=l_,
            n_cpus=cf["prepare_ds_virus"]["n_cpus"],
            random_seed=cf["prepare_ds_virus"]["random_seed"],
        )
        print(f"finished dataset preparation for {l_} fragment size")
    print(f"NN and RF datasets are stored in {cf['prepare_ds_virus']['virus_out_path']}")

def prepare_ds_eucaryotic(config):
    with open(config, "r") as jsonfile:
        cf = json.load(jsonfile)

    assert Path(cf["prepare_ds_eucaryotic"]["path_virus"]).exists(), f'{cf["prepare_ds_eucaryotic"]["path_virus"]} does not exist'
    assert Path(cf["prepare_ds_eucaryotic"]["path_eucaryotic"]).exists(), f'{cf["prepare_ds_eucaryotic"]["path_eucaryotic"]} does not exist'
    assert Path(cf["prepare_ds_eucaryotic"]["path_bact"]).exists(), f'{cf["prepare_ds_eucaryotic"]["path_bact"]} does not exist'

    Path(cf["prepare_ds_eucaryotic"]["eucaryotic_out_path"]).mkdir(parents=True, exist_ok=True)

    for l_ in 500, 1000:
        prepare_ds_nn(
            path_virus=cf["prepare_ds_eucaryotic"]["path_virus"],
            path_eucaryotic=cf["prepare_ds_eucaryotic"]["path_eucaryotic"],
            path_bact=cf["prepare_ds_eucaryotic"]["path_bact"],
            out_path=cf["prepare_ds_eucaryotic"]["eucaryotic_out_path"],
            fragment_length=l_,
            n_cpus=cf["prepare_ds_eucaryotic"]["n_cpus"],
            random_seed=cf["prepare_ds_eucaryotic"]["random_seed"],
        )
        prepare_ds_rf_eucaryotic(
            path_virus=cf["prepare_ds_eucaryotic"]["path_virus"],
            path_eucaryotic=cf["prepare_ds_eucaryotic"]["path_eucaryotic"],
            path_bact=cf["prepare_ds_eucaryotic"]["path_bact"],
            virus_out_path=cf["prepare_ds_virus"]["virus_out_path"],
            eucaryotic_out_path=cf["prepare_ds_eucaryotic"]["eucaryotic_out_path"],
            bacteria_out_path=cf["prepare_ds_bacteria"]["bacteria_out_path"],
            # out_path=cf["prepare_ds_eucaryotic"]["eucaryotic_out_path"],
            fragment_length=l_,
            n_cpus=cf["prepare_ds_eucaryotic"]["n_cpus"],
            random_seed=cf["prepare_ds_eucaryotic"]["random_seed"],
        )
        print(f"finished dataset preparation for {l_} fragment size")
    print(f"NN and RF datasets are stored in {cf['prepare_ds_eucaryotic']['eucaryotic_out_path']}")    

def prepare_ds_bacteria(config):
    with open(config, "r") as jsonfile:
        cf = json.load(jsonfile)

    assert Path(cf["prepare_ds_bacteria"]["path_virus"]).exists(), f'{cf["prepare_ds_bacteria"]["path_virus"]} does not exist'
    assert Path(cf["prepare_ds_bacteria"]["path_eucaryotic"]).exists(), f'{cf["prepare_ds_bacteria"]["path_eucaryotic"]} does not exist'
    assert Path(cf["prepare_ds_bacteria"]["path_bact"]).exists(), f'{cf["prepare_ds_bacteria"]["path_bact"]} does not exist'

    Path(cf["prepare_ds_bacteria"]["bacteria_out_path"]).mkdir(parents=True, exist_ok=True)

    for l_ in 500, 1000:
        prepare_ds_nn(
            path_virus=cf["prepare_ds_bacteria"]["path_virus"],
            path_eucaryotic=cf["prepare_ds_bacteria"]["path_eucaryotic"],
            path_bact=cf["prepare_ds_bacteria"]["path_bact"],
            out_path=cf["prepare_ds_bacteria"]["bacteria_out_path"],
            fragment_length=l_,
            n_cpus=cf["prepare_ds_bacteria"]["n_cpus"],
            random_seed=cf["prepare_ds_bacteria"]["random_seed"],
        )
        prepare_ds_rf_bacteria(
            path_virus=cf["prepare_ds_bacteria"]["path_virus"],
            path_eucaryotic=cf["prepare_ds_bacteria"]["path_eucaryotic"],
            path_bact=cf["prepare_ds_bacteria"]["path_bact"],
            # out_path=cf["prepare_ds_bacteria"]["eucaryotic_out_path"]
            virus_out_path=cf["prepare_ds_virus"]["virus_out_path"],
            eucaryotic_out_path=cf["prepare_ds_eucaryotic"]["eucaryotic_out_path"],
            bacteria_out_path=cf["prepare_ds_bacteria"]["bacteria_out_path"],
            fragment_length=l_,
            n_cpus=cf["prepare_ds_bacteria"]["n_cpus"],
            random_seed=cf["prepare_ds_bacteria"]["random_seed"],
        )
        print(f"finished dataset preparation for {l_} fragment size")
    print(f"NN and RF datasets are stored in {cf['prepare_ds_bacteria']['bacteria_out_path']}")    


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    prepare_ds_virus(config_file)
    prepare_ds_eucaryotic(config_file)
    prepare_ds_bacteria(config_file)
