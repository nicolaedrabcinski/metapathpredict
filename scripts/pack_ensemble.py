"""
Pack several scripts/baselines.py checkpoints (usually different seeds of the same recipe) into one
ensemble checkpoint that `metapathpredict evaluate`/`predict` and every script in this repo can load
like a single model (metapathpredict.baselines.EnsembleClassifier averages softmax probabilities).

    python scripts/pack_ensemble.py \
        experiments/baselines/final_s1_seed0/model.pt \
        experiments/baselines/final_s1_seed1/model.pt \
        experiments/baselines/final_s1_seed2/model.pt \
        --out experiments/baselines/final_ensemble_s1/ensemble.pt

To confirm this reproduces the reported ensemble numbers:
    metapathpredict evaluate --model experiments/baselines/final_ensemble_s1/ensemble.pt \
        --data data/datasets/taxa8vir_fam500_s1/encoded_test_500.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from metapathpredict.baselines import save_ensemble_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="model.pt files from scripts/baselines.py supervised --save-checkpoint")
    ap.add_argument("--out", required=True, help="output ensemble checkpoint path")
    args = ap.parse_args()

    configs, states, class_names = [], [], None
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" not in ckpt:
            raise SystemExit(f"{path}: not a scripts/baselines.py checkpoint (no 'state_dict' key)")
        states.append(ckpt["state_dict"])
        configs.append(ckpt["config"])
        names = ckpt["config"].get("class_names")
        if names:
            if class_names is not None and names != class_names:
                raise SystemExit(f"{path}: class_names {names} disagree with the earlier checkpoints' {class_names}")
            class_names = names

    # save_ensemble_checkpoint takes built modules, not raw state dicts, so it can validate the
    # architecture (via load_state_dict's strict check) before anything is written to disk.
    from metapathpredict.baselines import build_classifier

    models = []
    for state, config in zip(states, configs):
        model = build_classifier(config["num_classes"], config["backbone"], config["base_channels"],
                                 config.get("norm", "batch"), config.get("pool", "avg"), config.get("rc_share", "none"))
        model.load_state_dict(state)
        models.append(model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_ensemble_checkpoint(models, out, configs, class_names)
    print(f"{out}: {len(models)} members, {configs[0]['num_classes']} classes")


if __name__ == "__main__":
    main()
