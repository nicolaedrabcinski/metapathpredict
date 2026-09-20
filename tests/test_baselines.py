"""k-mer features and the cross-entropy CNN baseline."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from metapathpredict.baselines import (
    evaluate_accuracy,
    kmer_frequencies,
    load_backbone_weights,
    load_supervised_checkpoint,
    predict_probabilities,
    WithAuxTargets,
    save_supervised_checkpoint,
    train_supervised,
)
from metapathpredict.models.configurable_cnn import ConfigurableCNN
from metapathpredict.models.contrastive import ContrastiveEncoder


def _onehot(seq: str) -> np.ndarray:
    x = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in "ACGT":
            x["ACGT".index(base), i] = 1.0
    return x


def test_kmer_frequencies_match_a_hand_count():
    freq = kmer_frequencies(_onehot("ACGTAC")[None], k=2)[0]
    index = lambda kmer: "ACGT".index(kmer[0]) * 4 + "ACGT".index(kmer[1])
    assert freq[index("AC")] == pytest.approx(2 / 5)
    for kmer in ("CG", "GT", "TA"):
        assert freq[index(kmer)] == pytest.approx(1 / 5)
    assert freq.sum() == pytest.approx(1.0)


def test_windows_with_n_are_skipped_and_all_n_gives_zeros():
    freq = kmer_frequencies(np.stack([_onehot("ACNGT"), _onehot("NNNNN")]), k=2)
    assert freq[0].sum() == pytest.approx(1.0) and freq[0][0 * 4 + 1] == pytest.approx(0.5)  # AC and GT remain
    assert freq[1].sum() == 0.0


def test_kmer_chunking_gives_the_same_result():
    x = np.stack([_onehot("".join(np.random.default_rng(i).choice(list("ACGT"), 40))) for i in range(7)])
    assert np.allclose(kmer_frequencies(x, k=3, chunk=3), kmer_frequencies(x, k=3, chunk=100))


def _gc_data(n=256, length=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    probs = torch.tensor([[0.4, 0.1, 0.1, 0.4], [0.1, 0.4, 0.4, 0.1]])[y]  # AT-rich vs GC-rich, invariant to RC
    idx = torch.multinomial(probs.repeat_interleave(length, 0), 1, generator=g).view(n, length)
    return torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float(), y


@pytest.mark.parametrize("augment", ["none", "rc", "full"])
def test_supervised_baseline_learns_a_separable_problem(augment):
    torch.manual_seed(0)
    x, y = _gc_data()
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(*_gc_data(seed=1)), batch_size=64)
    model = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16)
    out = train_supervised(model, loader, val, "cpu", epochs=6, patience=0, lr=3e-3, augment=augment)
    assert out["best_val_acc"] > 0.9
    assert evaluate_accuracy(model, val, "cpu")[0] == pytest.approx(out["best_val_acc"])  # best weights restored
    assert len(out["history"]) == 6


def test_early_stopping_and_bad_augment_mode():
    torch.manual_seed(0)
    x, y = _gc_data()
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    model = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16)
    out = train_supervised(model, loader, loader, "cpu", epochs=30, patience=2, lr=1e-9)  # never improves after epoch 1
    assert len(out["history"]) < 30
    with pytest.raises(ValueError):
        train_supervised(model, loader, loader, "cpu", epochs=1, augment="bogus")


def _contrastive_checkpoint(tmp_path, base_channels=16):
    encoder = ContrastiveEncoder(backbone="small", projection_dim=32, hidden_dim=64,
                                 base_channels=base_channels, num_classes=2)
    path = tmp_path / "c.pt"
    torch.save({"encoder_state_dict": encoder.state_dict()}, path)
    return encoder, path


def test_backbone_weights_are_loaded_and_the_head_stays_new(tmp_path):
    encoder, path = _contrastive_checkpoint(tmp_path)
    torch.manual_seed(1)
    model = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16)
    head_before = {k: v.clone() for k, v in model.classifier.state_dict().items()}
    assert load_backbone_weights(model, path) > 0
    for (name, want), (_, got) in zip(encoder.encoder.features.state_dict().items(), model.features.state_dict().items()):
        assert torch.equal(want, got), name
    for name, value in model.classifier.state_dict().items():
        assert torch.equal(value, head_before[name])  # the contrastive head was not copied


def test_backbone_mismatch_is_an_error_not_a_partial_load(tmp_path):
    _, path = _contrastive_checkpoint(tmp_path, base_channels=16)
    with pytest.raises(Exception):
        load_backbone_weights(ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=32), path)


def test_probabilities_sum_to_one_and_agree_with_the_argmax_accuracy():
    torch.manual_seed(0)
    x, y = _gc_data()
    loader = DataLoader(TensorDataset(x, y), batch_size=64)
    model = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16)
    targets, probs = predict_probabilities(model, loader, "cpu")
    assert probs.shape == (len(y), 2) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs.argmax(axis=1) == targets).mean() == pytest.approx(evaluate_accuracy(model, loader, "cpu")[0])


def test_checkpoint_round_trip_rebuilds_the_same_model(tmp_path):
    torch.manual_seed(0)
    x, y = _gc_data(n=64)
    model = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16, norm="group").eval()
    path = tmp_path / "model.pt"
    save_supervised_checkpoint(model, path, {"backbone": "small", "base_channels": 16, "norm": "group", "num_classes": 2})
    loaded = load_supervised_checkpoint(path)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)


def test_cosine_schedule_anneals_the_learning_rate_and_bad_names_are_rejected(monkeypatch):
    created = []

    class Spy(torch.optim.AdamW):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(torch.optim, "AdamW", Spy)
    torch.manual_seed(0)
    x, y = _gc_data(n=64)
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    model = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16)
    train_supervised(model, loader, loader, "cpu", epochs=4, patience=0, lr=1e-2, lr_schedule="cosine")
    assert created[-1].param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)  # annealed to zero over the run
    train_supervised(model, loader, loader, "cpu", epochs=2, patience=0, lr=1e-2)
    assert created[-1].param_groups[0]["lr"] == pytest.approx(1e-2)  # constant by default
    with pytest.raises(ValueError):
        train_supervised(model, loader, loader, "cpu", epochs=1, lr_schedule="linear")


def test_auxiliary_head_is_trained_and_unlabelled_items_are_ignored():
    torch.manual_seed(0)
    x, y = _gc_data(n=128)
    aux = torch.where(torch.arange(len(y)) % 4 == 0, torch.full_like(y, -100), y * 2 + (torch.arange(len(y)) % 2))  # 4 aux classes
    loader = DataLoader(WithAuxTargets(TensorDataset(x, y), aux.numpy()), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(*_gc_data(seed=1)), batch_size=64)
    model = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16)
    head = torch.nn.Linear(model._final_channels, 4)
    before = head.weight.detach().clone()
    out = train_supervised(model, loader, val, "cpu", epochs=4, patience=0, lr=3e-3, augment="none", aux_head=head, aux_weight=0.5)
    assert not torch.equal(before, head.weight.detach())   # the auxiliary head took part in the optimisation
    assert out["best_val_acc"] > 0.8                       # and the main task still works


def test_without_an_auxiliary_head_the_batches_may_carry_extra_labels():
    x, y = _gc_data(n=64)
    loader = DataLoader(WithAuxTargets(TensorDataset(x, y), np.zeros(64, dtype=int)), batch_size=32)
    model = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16)
    train_supervised(model, loader, DataLoader(TensorDataset(x, y), batch_size=32), "cpu", epochs=1, patience=0)


def test_aux_targets_must_match_the_dataset_length():
    x, y = _gc_data(n=10)
    with pytest.raises(ValueError):
        WithAuxTargets(TensorDataset(x, y), np.zeros(9, dtype=int))


# ------------------------------------------------------------------ pooling and strand sharing
from metapathpredict.baselines import build_classifier  # noqa: E402
from metapathpredict.models.base import reverse_complement  # noqa: E402
from metapathpredict.models.configurable_cnn import RCShared  # noqa: E402


@pytest.mark.parametrize("pool,factor", [("avg", 1), ("max", 1), ("avgmax", 2)])
@pytest.mark.parametrize("preset", ["small", "multi"])
def test_pooling_modes_have_consistent_embedding_and_classifier_sizes(pool, factor, preset):
    model = ConfigurableCNN(num_classes=3, kernel_preset=preset, base_channels=16, pool=pool).eval()
    x = _gc_data(n=8)[0]
    embedding = model.get_embeddings(x)
    assert embedding.shape == (8, model._final_channels) and model.classifier[1].in_features == embedding.shape[1]
    assert model(x).shape == (8, 3)
    assert embedding.shape[1] == (model._final_channels // factor) * factor


def test_max_pool_differs_from_average_pool_and_bad_names_are_rejected():
    torch.manual_seed(0)
    x = _gc_data(n=8)[0]
    avg = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16, pool="avg").eval()
    mx = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16, pool="max").eval()
    mx.features.load_state_dict(avg.features.state_dict())
    assert (mx.get_embeddings(x) >= avg.get_embeddings(x) - 1e-6).all()      # a maximum is never below the mean
    assert not torch.allclose(mx.get_embeddings(x), avg.get_embeddings(x))
    with pytest.raises(ValueError):
        ConfigurableCNN(num_classes=2, pool="median")


@pytest.mark.parametrize("mode", ["mean", "max"])
def test_strand_sharing_gives_the_same_prediction_for_a_sequence_and_its_reverse_complement(mode):
    torch.manual_seed(0)
    x = _gc_data(n=8)[0]
    plain = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16).eval()
    shared = RCShared(plain, mode).eval()
    assert torch.allclose(shared(x), shared(reverse_complement(x)), atol=1e-5)
    assert not torch.allclose(plain(x), plain(reverse_complement(x)), atol=1e-5)  # the plain CNN is not invariant
    assert sum(p.numel() for p in shared.parameters()) == sum(p.numel() for p in plain.parameters())  # no extra weights
    with pytest.raises(ValueError):
        RCShared(plain, "sum")


def test_checkpoint_round_trip_keeps_pool_and_strand_sharing(tmp_path):
    torch.manual_seed(0)
    x = _gc_data(n=16)[0]
    model = build_classifier(2, "small", 16, "batch", "avgmax", "mean").eval()
    config = {"backbone": "small", "base_channels": 16, "norm": "batch", "pool": "avgmax", "rc_share": "mean", "num_classes": 2}
    save_supervised_checkpoint(model, tmp_path / "m.pt", config)
    loaded = load_supervised_checkpoint(tmp_path / "m.pt")
    assert isinstance(loaded, RCShared) and loaded.base.pool_type == "avgmax"
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)


def test_old_checkpoints_without_the_new_fields_still_load(tmp_path):
    model = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16).eval()
    torch.save({"state_dict": model.state_dict(), "config": {"backbone": "small", "base_channels": 16, "num_classes": 2}}, tmp_path / "old.pt")
    assert type(load_supervised_checkpoint(tmp_path / "old.pt")).__name__ == "ConfigurableCNN"


def test_a_strand_sharing_model_trains_and_takes_an_auxiliary_head():
    torch.manual_seed(0)
    x, y = _gc_data(n=128)
    loader = DataLoader(WithAuxTargets(TensorDataset(x, y), (y * 2).numpy()), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(*_gc_data(seed=1)), batch_size=64)
    model = build_classifier(2, "small", 16, "batch", "avgmax", "mean")
    head = torch.nn.Linear(model._final_channels, 4)
    out = train_supervised(model, loader, val, "cpu", epochs=4, patience=0, lr=3e-3, augment="none", aux_head=head)
    assert out["best_val_acc"] > 0.8


# ------------------------------------------------------------------ ensemble checkpoints (Q-1)
from metapathpredict.baselines import (  # noqa: E402
    EnsembleClassifier,
    load_ensemble_checkpoint,
    save_ensemble_checkpoint,
)


def test_ensemble_averages_softmax_probabilities_not_logits():
    torch.manual_seed(0)
    x = _gc_data(n=8)[0]
    members = [ConfigurableCNN(num_classes=3, kernel_preset="small", base_channels=16).eval() for _ in range(3)]
    ensemble = EnsembleClassifier(members).eval()
    with torch.no_grad():
        expected = torch.stack([torch.softmax(m(x), dim=1) for m in members]).mean(dim=0)
        got = torch.softmax(ensemble(x), dim=1)  # forward() returns log-probs, softmax undoes the log
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(got.sum(dim=1), torch.ones(8), atol=1e-5)


def test_ensemble_of_one_well_trained_model_reproduces_its_own_accuracy():
    torch.manual_seed(0)
    x, y = _gc_data(n=64)
    good = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=16)
    train_supervised(good, loader, loader, "cpu", epochs=15, patience=0, lr=5e-3, augment="none")
    ensemble = EnsembleClassifier([good]).eval()
    with torch.no_grad():
        single_preds = good(x).argmax(dim=1)
        ensemble_preds = ensemble(x).argmax(dim=1)
    assert torch.equal(single_preds, ensemble_preds)
    assert (single_preds == y).float().mean() > 0.85


def test_ensemble_of_three_identical_copies_matches_the_shared_accuracy():
    torch.manual_seed(0)
    x, y = _gc_data(n=64)
    good = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=16)
    train_supervised(good, loader, loader, "cpu", epochs=15, patience=0, lr=5e-3, augment="none")
    ensemble = EnsembleClassifier([good, good, good]).eval()  # same weights three times: averaging changes nothing
    with torch.no_grad():
        assert torch.equal(good(x).argmax(dim=1), ensemble(x).argmax(dim=1))


def test_ensemble_requires_at_least_one_member():
    with pytest.raises(ValueError):
        EnsembleClassifier([])


def test_save_and_load_ensemble_checkpoint_roundtrips(tmp_path):
    torch.manual_seed(0)
    x = _gc_data(n=8)[0]
    members = [ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16).eval() for _ in range(2)]
    configs = [{"backbone": "small", "base_channels": 16, "num_classes": 2} for _ in members]
    path = tmp_path / "ensemble.pt"
    save_ensemble_checkpoint(members, path, configs, class_names=["a", "b"])

    with pytest.raises(ValueError):
        save_ensemble_checkpoint(members, path, configs[:1])  # mismatched lengths

    loaded = load_ensemble_checkpoint(path)
    assert isinstance(loaded, EnsembleClassifier) and len(loaded.members) == 2
    with torch.no_grad():
        expected = torch.stack([torch.softmax(m(x), dim=1) for m in members]).mean(dim=0)
        got = torch.softmax(loaded(x), dim=1)
    assert torch.allclose(got, expected, atol=1e-6)


def test_ensemble_checkpoint_loads_through_the_cli_like_a_single_model(tmp_path):
    from metapathpredict.cli import _load_model_from_checkpoint, _predict_single

    torch.manual_seed(0)
    x = _gc_data(n=4)[0]
    members = [build_classifier(3, "small", 16, "batch", "avg", "mean") for _ in range(2)]
    configs = [{"backbone": "small", "base_channels": 16, "num_classes": 3, "norm": "batch", "pool": "avg", "rc_share": "mean"}
              for _ in members]
    path = tmp_path / "ensemble.pt"
    save_ensemble_checkpoint(members, path, configs, class_names=["bacteria", "archaea", "virus"])

    loaded, model_type = _load_model_from_checkpoint(path, torch.device("cpu"))
    assert model_type == "supervised" and loaded.class_names == ["bacteria", "archaea", "virus"]
    class_idx, confidence, probs = _predict_single(loaded, model_type, x[:1])
    assert len(probs) == 3 and confidence == pytest.approx(max(probs), abs=1e-6)
