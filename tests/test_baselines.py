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
