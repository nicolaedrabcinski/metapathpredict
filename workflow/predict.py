import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import tensorflow as tf
import numpy as np
from Bio import SeqIO
import pandas as pd
from utils import preprocess as pp
from pathlib import Path
from models import model_5, model_7, model_10
from joblib import load
import psutil


def predict_nn_virus(virus_ds_path, eucaryotic_ds_path, bacteria_ds_path, nn_weights_path, length, n_cpus=1, batch_size=256):
    """
    Breaks down contigs into fragments and uses pretrained neural networks to give predictions for fragments
    """

    try:
        pid = psutil.Process(os.getpid())
        pid.cpu_affinity(range(n_cpus))
    except AttributeError:
        print("CPU allocation is not working properly. This will not impact the analysis results but may increase the runtime")
    
    try:
        # seqs_ = list(SeqIO.parse(virus_ds_path, "fasta"))
        seqs_virus = list(SeqIO.parse(virus_ds_path, "fasta"))
        seqs_eucaryotic = list(SeqIO.parse(eucaryotic_ds_path, "fasta"))
        seqs_bacteria = list(SeqIO.parse(bacteria_ds_path, "fasta"))
    except FileNotFoundError:
        raise Exception("Test dataset was not found. Change dataset variable.")
    
    seqs_ = seqs_virus + seqs_eucaryotic + seqs_bacteria

    out_table = {
        "id": [],
        "length": [],
        "fragment": [],
        "pred_eucaryotic_5": [],
        "pred_vir_5": [],
        "pred_bact_5": [],
        "pred_eucaryotic_7": [],
        "pred_vir_7": [],
        "pred_bact_7": [],
        "pred_eucaryotic_10": [],
        "pred_vir_10": [],
        "pred_bact_10": [],
    }

    if not seqs_:
        raise ValueError("All sequences were smaller than length of the model")
    test_fragments = []
    test_fragments_rc = []
    
    for seq in seqs_:
        fragments_, fragments_rc, _ = pp.fragmenting([seq], length, max_gap=0.8, sl_wind_step=int(length / 2))
        test_fragments.extend(fragments_)
        test_fragments_rc.extend(fragments_rc)

        for j in range(len(fragments_)):
            out_table["id"].append(seq.id)
            out_table["length"].append(len(seq.seq))
            out_table["fragment"].append(j)

    test_encoded = np.concatenate([pp.one_hot_encode(s) for s in pp.chunks(test_fragments, int(len(test_fragments) / n_cpus + 1))])
    test_encoded_rc = np.concatenate([pp.one_hot_encode(s) for s in pp.chunks(test_fragments_rc, int(len(test_fragments_rc) / n_cpus + 1))])

    for model, s in zip([model_5.model(length), model_7.model(length), model_10.model(length)], [5, 7, 10]):
        model.load_weights(Path(nn_weights_path, f"/home/nicolaedrabcinski/eebg_project/data/weights/weights_viral/model_{s}_{length}.weights.h5"))
        prediction = model.predict([test_encoded, test_encoded_rc], batch_size)

        out_table[f"pred_eucaryotic_{s}"].extend(list(prediction[..., 0]))
        out_table[f"pred_vir_{s}"].extend(list(prediction[..., 1]))
        out_table[f"pred_bact_{s}"].extend(list(prediction[..., 2]))

    return pd.DataFrame(out_table).round(3)

def predict_nn_eucaryotic(virus_ds_path, eucaryotic_ds_path, bacteria_ds_path, nn_weights_path, length, n_cpus=1, batch_size=256):
    """
    Breaks down contigs into fragments
    and uses pretrained neural networks to give predictions for fragments
    """
    try:
        pid = psutil.Process(os.getpid())
        pid.cpu_affinity(range(n_cpus))
    except AttributeError:
        print("CPU allocation is not working properly. This will not impact the analysis results but may increase the runtime")

    try:
        # seqs_ = list(SeqIO.parse(eucaryotic_ds_path, "fasta"))
        seqs_virus = list(SeqIO.parse(virus_ds_path, "fasta"))
        seqs_eucaryotic = list(SeqIO.parse(eucaryotic_ds_path, "fasta"))
        seqs_bacteria = list(SeqIO.parse(bacteria_ds_path, "fasta"))
    except FileNotFoundError:
        raise Exception("Test dataset was not found. Change ds variable")

    seqs_ = seqs_virus + seqs_eucaryotic + seqs_bacteria

    out_table = {
        "id": [],
        "length": [],
        "fragment": [],
        "pred_eucaryotic_5": [],
        "pred_vir_5": [],
        "pred_bact_5": [],
        "pred_eucaryotic_7": [],
        "pred_vir_7": [],
        "pred_bact_7": [],
        "pred_eucaryotic_10": [],
        "pred_vir_10": [],
        "pred_bact_10": [],
    }

    if not seqs_:
        raise ValueError("All sequences were smaller than length of the model")
    
    test_fragments = []
    test_fragments_rc = []
    
    for seq in seqs_:
        fragments_, fragments_rc, _ = pp.fragmenting([seq], length, max_gap=0.8, sl_wind_step=int(length / 2))
        test_fragments.extend(fragments_)
        test_fragments_rc.extend(fragments_rc)

        for j in range(len(fragments_)):
            out_table["id"].append(seq.id)
            out_table["length"].append(len(seq.seq))
            out_table["fragment"].append(j)

    # Perform one-hot encoding without using Ray
    test_encoded = np.concatenate([pp.one_hot_encode(s) for s in pp.chunks(test_fragments, int(len(test_fragments) / n_cpus + 1))])
    test_encoded_rc = np.concatenate([pp.one_hot_encode(s) for s in pp.chunks(test_fragments_rc, int(len(test_fragments_rc) / n_cpus + 1))])

    for model, s in zip([model_5.model(length), model_7.model(length), model_10.model(length)], [5, 7, 10]):
        model.load_weights(Path(nn_weights_path, f"/home/nicolaedrabcinski/eebg_project/data/weights/weights_eucaryotic/model_{s}_{length}.weights.h5"))
        prediction = model.predict([test_encoded, test_encoded_rc], batch_size)

        out_table[f"pred_eucaryotic_{s}"].extend(list(prediction[..., 0]))
        out_table[f"pred_vir_{s}"].extend(list(prediction[..., 1]))
        out_table[f"pred_bact_{s}"].extend(list(prediction[..., 2]))

    return pd.DataFrame(out_table).round(3)

def predict_nn_bacteria(virus_ds_path, eucaryotic_ds_path, bacteria_ds_path, nn_weights_path, length, n_cpus=1, batch_size=256):
    """
    Breaks down contigs into fragments
    and uses pretrained neural networks to give predictions for fragments
    """
    try:
        pid = psutil.Process(os.getpid())
        pid.cpu_affinity(range(n_cpus))
    except AttributeError:
        print("CPU allocation is not working properly. This will not impact the analysis results but may increase the runtime")

    try:
        # seqs_ = list(SeqIO.parse(bacteria_ds_path, "fasta"))
        seqs_virus = list(SeqIO.parse(virus_ds_path, "fasta"))
        seqs_eucaryotic = list(SeqIO.parse(eucaryotic_ds_path, "fasta"))
        seqs_bacteria = list(SeqIO.parse(bacteria_ds_path, "fasta"))
    except FileNotFoundError:
        raise Exception("Test dataset was not found. Change ds variable")

    seqs_ = seqs_virus + seqs_eucaryotic + seqs_bacteria

    out_table = {
        "id": [],
        "length": [],
        "fragment": [],
        "pred_eucaryotic_5": [],
        "pred_vir_5": [],
        "pred_bact_5": [],
        "pred_eucaryotic_7": [],
        "pred_vir_7": [],
        "pred_bact_7": [],
        "pred_eucaryotic_10": [],
        "pred_vir_10": [],
        "pred_bact_10": [],
    }
    if not seqs_:
        raise ValueError("All sequences were smaller than length of the model")
    test_fragments = []
    test_fragments_rc = []
    
    for seq in seqs_:
        fragments_, fragments_rc, _ = pp.fragmenting([seq], length, max_gap=0.8, sl_wind_step=int(length / 2))
        test_fragments.extend(fragments_)
        test_fragments_rc.extend(fragments_rc)
        for j in range(len(fragments_)):
            out_table["id"].append(seq.id)
            out_table["length"].append(len(seq.seq))
            out_table["fragment"].append(j)

    # Perform one-hot encoding without using Ray
    test_encoded = np.concatenate([pp.one_hot_encode(s) for s in pp.chunks(test_fragments, int(len(test_fragments) / n_cpus + 1))])
    test_encoded_rc = np.concatenate([pp.one_hot_encode(s) for s in pp.chunks(test_fragments_rc, int(len(test_fragments_rc) / n_cpus + 1))])

    for model, s in zip([model_5.model(length), model_7.model(length), model_10.model(length)], [5, 7, 10]):
        model.load_weights(Path(nn_weights_path, f"/home/nicolaedrabcinski/eebg_project/data/weights/weights_bacteria/model_{s}_{length}.weights.h5"))
        prediction = model.predict([test_encoded, test_encoded_rc], batch_size)
        out_table[f"pred_eucaryotic_{s}"].extend(list(prediction[..., 0]))
        out_table[f"pred_vir_{s}"].extend(list(prediction[..., 1]))
        out_table[f"pred_bact_{s}"].extend(list(prediction[..., 2]))
    return pd.DataFrame(out_table).round(3)

def predict_rf_virus(df, rf_weights_path, length):
    """
    Using predictions by predict_nn and weights of a trained RF classifier gives a single prediction for a fragment
    """
    clf = load(Path(rf_weights_path, f"RF_{length}.joblib"))
    
    X = df[
        ["pred_eucaryotic_5", "pred_vir_5", "pred_bact_5", "pred_eucaryotic_7", "pred_vir_7", "pred_bact_7", "pred_eucaryotic_10", "pred_vir_10", "pred_bact_10"]]
    
    y_pred = clf.predict(X)
    mapping = {0: "eucaryotic", 1: "virus", 2: "bacteria"}
    
    df["RF_decision"] = np.vectorize(mapping.get)(y_pred)
    prob_classes = clf.predict_proba(X)
    df["RF_pred_eucaryotic"] = prob_classes[..., 0]
    df["RF_pred_vir"] = prob_classes[..., 1]
    df["RF_pred_bact"] = prob_classes[..., 2]
    
    return df

def predict_rf_eucaryotic(df, rf_weights_path, length):
    """
    Using predictions by predict_nn and weights of a trained RF classifier gives a single prediction for a fragment
    """
    clf = load(Path(rf_weights_path, f"RF_{length}.joblib"))
    X = df[
        ["pred_eucaryotic_5", "pred_vir_5", "pred_bact_5", "pred_eucaryotic_7", "pred_vir_7", "pred_bact_7", "pred_eucaryotic_10", "pred_vir_10", "pred_bact_10"]]
    y_pred = clf.predict(X)
    mapping = {0: "eucaryotic", 1: "virus", 2: "bacteria"}
    df["RF_decision"] = np.vectorize(mapping.get)(y_pred)
    prob_classes = clf.predict_proba(X)
    df["RF_pred_eucaryotic"] = prob_classes[..., 0]
    df["RF_pred_vir"] = prob_classes[..., 1]
    df["RF_pred_bact"] = prob_classes[..., 2]
    return df

def predict_rf_bacteria(df, rf_weights_path, length):
    """
    Using predictions by predict_nn and weights of a trained RF classifier gives a single prediction for a fragment
    """
    clf = load(Path(rf_weights_path, f"RF_{length}.joblib"))
    
    X = df[
        ["pred_eucaryotic_5", "pred_vir_5", "pred_bact_5", "pred_eucaryotic_7", "pred_vir_7", "pred_bact_7", "pred_eucaryotic_10", "pred_vir_10", "pred_bact_10"]]
    
    y_pred = clf.predict(X)
    mapping = {0: "eucaryotic", 1: "virus", 2: "bacteria"}
    df["RF_decision"] = np.vectorize(mapping.get)(y_pred)
    prob_classes = clf.predict_proba(X)
    
    df["RF_pred_eucaryotic"] = prob_classes[..., 0]
    df["RF_pred_vir"] = prob_classes[..., 1]
    df["RF_pred_bact"] = prob_classes[..., 2]
    
    return df

def predict_contigs(df):
    """
    Based on predictions of predict_rf for fragments gives a final prediction for the whole contig
    """
    df = (
        df.groupby(["id", "length", 'RF_decision'], sort=False)
        .size()
        .unstack(fill_value=0)
    )

    df = df.reset_index()
    df = df.reindex(['id', 'length', 'virus', 'eucaryotic', 'bacteria'], axis=1).fillna(value=0)
    conditions = [
        (df['virus'] > df['eucaryotic']) & (df['virus'] > df['bacteria']),
        (df['eucaryotic'] > df['virus']) & (df['eucaryotic'] > df['bacteria']),
        (df['bacteria'] >= df['eucaryotic']) & (df['bacteria'] >= df['virus']),
    ]

    choices = ['virus', 'eucaryotic', 'bacteria']

    df['decision'] = np.select(conditions, choices, default='bacteria')
    df = df.loc[:, ['id', 'length', 'virus', 'eucaryotic', 'bacteria', 'decision']]
    df = df.rename(columns={'virus': '# viral fragments', 'bacteria': '# bacterial fragments', 'eucaryotic': '# eucaryotic fragments'})
    df['# viral / # total'] = (df['# viral fragments'] / (df['# viral fragments'] + df['# bacterial fragments'] + df['# eucaryotic fragments'])).round(3)
    df['# eucaryotic / # total'] = (df['# eucaryotic fragments'] / (df['# viral fragments'] + df['# bacterial fragments'] + df['# eucaryotic fragments'])).round(3)
    df['# bacteria / # total'] = (df['# bacterial fragments'] / (df['# viral fragments'] + df['# bacterial fragments'] + df['# eucaryotic fragments'])).round(3)
    
    df = df.sort_values(by='# viral fragments', ascending=False)
    
    return df

def predict_viral(config):
    """
    Predicts viral contigs from the fasta file
    Arguments:
    config file containing following fields:
        test_ds - path to the file with sequences for prediction (fasta format)
        weights - path to the folder with weights of pretrained NN and RF weights.
        This folder should contain two subfolders 500 and 1000. Each of them contains corresponding weight.
        out_folder - path to the folder, where to store output. You should create it
        return_viral - return contigs annotated as viral by virhunter (fasta format)
    """
    with open(config, "r") as jsonfile:
        cf = json.load(jsonfile)

    viruses_ds = cf["predict_viral"]["viruses_ds"]
    if isinstance(viruses_ds, list):
        pass
    elif isinstance(viruses_ds, str):
        test_ds = [viruses_ds]
    else:
        raise ValueError('test_ds was incorrectly assigned in the config file')

    assert Path(test_ds[0]).exists(), f'{test_ds[0]} does not exist'
    assert Path(cf["predict_viral"]["weights"]).exists(), f'{cf["predict_viral"]["weights"]} does not exist'
    assert isinstance(cf["predict_viral"]["limit"], int), 'limit should be an integer'
    Path(cf['predict_viral']['out_path']).mkdir(parents=True, exist_ok=True)

    for ts in test_ds:
        dfs_fr = []
        dfs_cont = []
        for l_ in 500, 1000:
            print(f'starting prediction for {Path(ts).name} for fragment length {l_}')
            df = predict_nn_virus(
                virus_ds_path=cf["predict_viral"]["viruses_ds"],
                eucaryotic_ds_path=cf["predict_viral"]["eucaryotic_ds"],
                bacteria_ds_path=cf["predict_viral"]["bacteria_ds"],
                nn_weights_path=cf["predict_viral"]["weights"],
                length=l_,
                n_cpus=cf["predict_viral"]["n_cpus"],
            )
            df = predict_rf_virus(
                df=df,
                rf_weights_path=cf["predict_viral"]["weights"],
                length=l_,
            )
            dfs_fr.append(df.round(3))
            df = predict_contigs(df).round(3)
            dfs_cont.append(df)
            print('prediction finished')
        limit = cf["predict_viral"]["limit"]
        df_500 = dfs_fr[0][(dfs_fr[0]['length'] >= limit) & (dfs_fr[0]['length'] < 1500)]
        df_1000 = dfs_fr[1][(dfs_fr[1]['length'] >= 1500)]
        df = pd.concat([df_1000, df_500], ignore_index=True)
        pred_fr = Path(cf['predict_viral']['out_path'], f"{Path(ts).stem}_predicted_fragments.csv")
        df.to_csv(pred_fr)

        df_500 = dfs_cont[0][(dfs_cont[0]['length'] >= limit) & (dfs_cont[0]['length'] < 1500)]
        df_1000 = dfs_cont[1][(dfs_cont[1]['length'] >= 1500)]
        df = pd.concat([df_1000, df_500], ignore_index=True)
        pred_contigs = Path(cf['predict_viral']['out_path'], f"{Path(ts).stem}_predicted.csv")
        df.to_csv(pred_contigs)

        if cf["predict_viral"]["return_viral"]:
            viral_ids = list(df[df["decision"] == "virus"]["id"])
            viral_seqs_ = list(SeqIO.parse(ts, "fasta"))
            viral_seqs = [s_ for s_ in viral_seqs_ if s_.id in viral_ids]
            SeqIO.write(viral_seqs, Path(cf['predict_viral']['out_path'], f"{Path(ts).stem}_viral.fasta"), 'fasta')

def predict_eucaryotic(config):
    """
    Predicts eucaryotic contigs from the fasta file
    Arguments:
    config file containing following fields:
        test_ds - path to the file with sequences for prediction (fasta format)
        weights - path to the folder with weights of pretrained NN and RF weights.
        This folder should contain two subfolders 500 and 1000. Each of them contains corresponding weight.
        out_folder - path to the folder, where to store output. You should create it
        return_eucaryotic - return contigs annotated as viral by virhunter (fasta format)
    """
    with open(config, "r") as jsonfile:
        cf = json.load(jsonfile)

    eucaryotic_ds = cf["predict_eucaryotic"]["eucaryotic_ds"]
    if isinstance(eucaryotic_ds, list):
        pass
    elif isinstance(eucaryotic_ds, str):
        test_ds = [eucaryotic_ds]
    else:
        raise ValueError('test_ds was incorrectly assigned in the config file')

    assert Path(test_ds[0]).exists(), f'{test_ds[0]} does not exist'
    assert Path(cf["predict_eucaryotic"]["weights"]).exists(), f'{cf["predict_eucaryotic"]["weights"]} does not exist'
    assert isinstance(cf["predict_eucaryotic"]["limit"], int), 'limit should be an integer'
    Path(cf['predict_eucaryotic']['out_path']).mkdir(parents=True, exist_ok=True)

    for ts in test_ds:
        dfs_fr = []
        dfs_cont = []
        for l_ in 500, 1000:
            print(f'starting prediction for {Path(ts).name} for fragment length {l_}')
            df = predict_nn_eucaryotic(
                virus_ds_path=cf["predict_eucaryotic"]["viruses_ds"],
                eucaryotic_ds_path=cf["predict_eucaryotic"]["eucaryotic_ds"],
                bacteria_ds_path=cf["predict_eucaryotic"]["bacteria_ds"],
                # eucaryotic_ds_path=ts,
                nn_weights_path=cf["predict_eucaryotic"]["weights"],
                length=l_,
                n_cpus=cf["predict_eucaryotic"]["n_cpus"],
            )
            df = predict_rf_eucaryotic(
                df=df,
                rf_weights_path=cf["predict_eucaryotic"]["weights"],
                length=l_,
            )
            dfs_fr.append(df.round(3))
            df = predict_contigs(df).round(3)
            dfs_cont.append(df)
            print('prediction finished')
        limit = cf["predict_eucaryotic"]["limit"]
        df_500 = dfs_fr[0][(dfs_fr[0]['length'] >= limit) & (dfs_fr[0]['length'] < 1500)]
        df_1000 = dfs_fr[1][(dfs_fr[1]['length'] >= 1500)]
        df = pd.concat([df_1000, df_500], ignore_index=True)
        pred_fr = Path(cf['predict_eucaryotic']['out_path'], f"{Path(ts).stem}_predicted_fragments.csv")
        df.to_csv(pred_fr)

        df_500 = dfs_cont[0][(dfs_cont[0]['length'] >= limit) & (dfs_cont[0]['length'] < 1500)]
        df_1000 = dfs_cont[1][(dfs_cont[1]['length'] >= 1500)]
        df = pd.concat([df_1000, df_500], ignore_index=True)
        pred_contigs = Path(cf['predict_eucaryotic']['out_path'], f"{Path(ts).stem}_predicted.csv")
        df.to_csv(pred_contigs)

        if cf["predict_eucaryotic"]["return_eucaryotic"]:
            eucaryotic_ids = list(df[df["decision"] == "eucaryotic"]["id"])
            eucaryotic_seqs_ = list(SeqIO.parse(ts, "fasta"))
            eucaryotic_seqs = [s_ for s_ in eucaryotic_seqs_ if s_.id in eucaryotic_ids]
            SeqIO.write(eucaryotic_seqs, Path(cf['predict_eucaryotic']['out_path'], f"{Path(ts).stem}_eucaryotic.fasta"), 'fasta')

def predict_bacteria(config):
    """
    Predicts bacteria contigs from the fasta file
    Arguments:
    config file containing following fields:
        test_ds - path to the file with sequences for prediction (fasta format)
        weights - path to the folder with weights of pretrained NN and RF weights.
        This folder should contain two subfolders 500 and 1000. Each of them contains corresponding weight.
        out_folder - path to the folder, where to store output. You should create it
        return_bacteria - return contigs annotated as bacteria by virhunter (fasta format)
    """
    with open(config, "r") as jsonfile:
        cf = json.load(jsonfile)

    bacteria_ds = cf["predict_bacteria"]["bacteria_ds"]
    if isinstance(bacteria_ds, list):
        pass
    elif isinstance(bacteria_ds, str):
        test_ds = [bacteria_ds]
    else:
        raise ValueError('test_ds was incorrectly assigned in the config file')

    assert Path(test_ds[0]).exists(), f'{test_ds[0]} does not exist'
    assert Path(cf["predict_bacteria"]["weights"]).exists(), f'{cf["predict_bacteria"]["weights"]} does not exist'
    assert isinstance(cf["predict_bacteria"]["limit"], int), 'limit should be an integer'
    Path(cf['predict_bacteria']['out_path']).mkdir(parents=True, exist_ok=True)

    for ts in test_ds:
        dfs_fr = []
        dfs_cont = []
        for l_ in 500, 1000:
            print(f'starting prediction for {Path(ts).name} for fragment length {l_}')
            df = predict_nn_bacteria(
                bacteria_ds_path=cf["predict_bacteria"]["bacteria_ds"],
                eucaryotic_ds_path=cf["predict_bacteria"]["eucaryotic_ds"],
                virus_ds_path=cf["predict_bacteria"]["viruses_ds"],
                nn_weights_path=cf["predict_bacteria"]["weights"],
                length=l_,
                n_cpus=cf["predict_bacteria"]["n_cpus"],
            )
            df = predict_rf_bacteria(
                df=df,
                rf_weights_path=cf["predict_bacteria"]["weights"],
                length=l_,
            )
            dfs_fr.append(df.round(3))
            df = predict_contigs(df).round(3)
            dfs_cont.append(df)
            print('prediction finished')
        limit = cf["predict_bacteria"]["limit"]
        df_500 = dfs_fr[0][(dfs_fr[0]['length'] >= limit) & (dfs_fr[0]['length'] < 1500)]
        df_1000 = dfs_fr[1][(dfs_fr[1]['length'] >= 1500)]
        df = pd.concat([df_1000, df_500], ignore_index=True)
        pred_fr = Path(cf['predict_bacteria']['out_path'], f"{Path(ts).stem}_predicted_fragments.csv")
        df.to_csv(pred_fr)

        df_500 = dfs_cont[0][(dfs_cont[0]['length'] >= limit) & (dfs_cont[0]['length'] < 1500)]
        df_1000 = dfs_cont[1][(dfs_cont[1]['length'] >= 1500)]
        df = pd.concat([df_1000, df_500], ignore_index=True)
        pred_contigs = Path(cf['predict_bacteria']['out_path'], f"{Path(ts).stem}_predicted.csv")
        df.to_csv(pred_contigs)

        if cf["predict_bacteria"]["return_bacteria"]:
            bacteria_ids = list(df[df["decision"] == "eucaryotic"]["id"])
            bacteria_seqs_ = list(SeqIO.parse(ts, "fasta"))
            bacteria_seqs = [s_ for s_ in bacteria_seqs_ if s_.id in bacteria_ids]
            SeqIO.write(bacteria_seqs, Path(cf['predict_bacteria']['out_path'], f"{Path(ts).stem}_bacteria.fasta"), 'fasta')


if __name__ == '__main__':
    config_path = "/home/nicolaedrabcinski/eebg_project/main_config.json"  # Provide the path to your JSON configuration file
    predict_viral(config_path)
    predict_eucaryotic(config_path)
    predict_bacteria(config_path)