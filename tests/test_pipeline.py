"""
Tests for the full Contrastive + RL pipeline integration.
"""

import argparse
import csv
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from metapathpredict.models.contrastive import (
    ContrastiveAugmentation,
    ContrastiveEncoder,
    ContrastiveTrainer,
)
from metapathpredict.models.reinforcement import (
    ActorCriticAgent,
    DQNAgent,
    PolicyGradientAgent,
    RLTrainer,
    SequenceEnvironment,
)


@pytest.fixture
def dummy_data():
    """Create small dummy dataset for testing."""
    sequences = torch.randn(32, 4, 500)
    labels = torch.randint(0, 3, (32,))
    return sequences, labels


@pytest.fixture
def dummy_dataloader(dummy_data):
    sequences, labels = dummy_data
    dataset = TensorDataset(sequences, labels)
    return DataLoader(dataset, batch_size=8)


class TestWeightTransfer:
    """Test contrastive encoder -> RL agent weight transfer."""

    def test_encoder_keys_match_agent(self):
        """Verify that ContrastiveEncoder.encoder keys are a subset of RL agent keys."""
        encoder = ContrastiveEncoder(in_channels=4, backbone="small")
        agent = ActorCriticAgent(in_channels=4, num_actions=3, backbone="small")

        enc_keys = {k for k in encoder.state_dict() if k.startswith("encoder.")}
        agent_keys = set(agent.state_dict().keys())

        # All encoder.* keys from ContrastiveEncoder should exist in agent
        overlap = enc_keys & agent_keys
        assert len(overlap) > 0, "No shared keys between encoder and agent"
        assert overlap == enc_keys, f"Missing keys in agent: {enc_keys - agent_keys}"

    @pytest.mark.parametrize("agent_cls", [DQNAgent, PolicyGradientAgent, ActorCriticAgent])
    def test_transfer_to_all_agents(self, agent_cls):
        """Test weight transfer works for all RL agent types."""
        encoder = ContrastiveEncoder(in_channels=4, backbone="small")
        agent = agent_cls(in_channels=4, num_actions=3, backbone="small")

        sd = encoder.state_dict()
        transfer = {k: v for k, v in sd.items() if k.startswith("encoder.")}

        missing, unexpected = agent.load_state_dict(transfer, strict=False)

        assert len(transfer) > 0
        assert len(unexpected) == 0
        # Missing keys should be agent-specific heads only
        for k in missing:
            assert not k.startswith("encoder."), f"Encoder key missing: {k}"

    def test_transfer_via_checkpoint(self, tmp_path):
        """Test full save/load/transfer cycle."""
        encoder = ContrastiveEncoder(in_channels=4, backbone="small")

        # Save contrastive checkpoint
        ckpt_path = tmp_path / "contrastive_best.pt"
        torch.save({
            "encoder_state_dict": encoder.state_dict(),
            "config": {"backbone": "small"},
        }, ckpt_path)

        # Load into RL agent
        agent = ActorCriticAgent(in_channels=4, num_actions=3, backbone="small")
        ckpt = torch.load(ckpt_path, weights_only=False)
        sd = ckpt["encoder_state_dict"]
        transfer = {k: v for k, v in sd.items() if k.startswith("encoder.")}
        agent.load_state_dict(transfer, strict=False)

        # Verify weights match
        for key in transfer:
            torch.testing.assert_close(
                agent.state_dict()[key],
                encoder.state_dict()[key],
            )


class TestFullPipeline:
    """Test the full contrastive -> RL pipeline."""

    def test_contrastive_training(self, dummy_dataloader):
        """Test contrastive training produces valid checkpoint."""
        encoder = ContrastiveEncoder(in_channels=4, backbone="small")
        aug = ContrastiveAugmentation(mutation_rate=0.1, mask_rate=0.15)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)

        trainer = ContrastiveTrainer(
            encoder=encoder,
            optimizer=optimizer,
            augmentation=aug,
            temperature=0.07,
            use_supervised=True,
            device="cpu",
        )

        loss = trainer.train_epoch(dummy_dataloader)
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))

    def test_rl_training(self, dummy_data):
        """Test RL training produces valid metrics."""
        sequences, labels = dummy_data
        agent = ActorCriticAgent(in_channels=4, num_actions=3, backbone="small")
        env = SequenceEnvironment(sequences, labels)
        optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4)

        trainer = RLTrainer(
            agent=agent,
            environment=env,
            optimizer=optimizer,
            device="cpu",
            gamma=0.99,
            algorithm="actor_critic",
        )

        metrics = trainer.train_epoch(num_episodes=10, epsilon=0.0)
        assert "avg_reward" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_full_pipeline_end_to_end(self, dummy_data, dummy_dataloader, tmp_path):
        """Test complete pipeline: contrastive pretrain -> save -> load -> RL fine-tune."""
        sequences, labels = dummy_data

        # Phase 1: Contrastive pretraining
        encoder = ContrastiveEncoder(in_channels=4, backbone="small")
        aug = ContrastiveAugmentation(mutation_rate=0.1, mask_rate=0.15)
        opt_c = torch.optim.AdamW(encoder.parameters(), lr=1e-3)

        trainer_c = ContrastiveTrainer(
            encoder=encoder, optimizer=opt_c, augmentation=aug,
            temperature=0.07, use_supervised=True, device="cpu",
        )

        loss = trainer_c.train_epoch(dummy_dataloader)

        # Save contrastive checkpoint
        ckpt_path = tmp_path / "contrastive_best.pt"
        torch.save({
            "encoder_state_dict": encoder.state_dict(),
            "config": {"backbone": "small"},
        }, ckpt_path)

        # Phase 2: RL fine-tuning with transferred weights
        agent = ActorCriticAgent(in_channels=4, num_actions=3, backbone="small")
        ckpt = torch.load(ckpt_path, weights_only=False)
        sd = ckpt["encoder_state_dict"]
        transfer = {k: v for k, v in sd.items() if k.startswith("encoder.")}
        agent.load_state_dict(transfer, strict=False)

        env = SequenceEnvironment(sequences, labels)
        opt_r = torch.optim.AdamW(agent.parameters(), lr=1e-4)

        trainer_r = RLTrainer(
            agent=agent, environment=env, optimizer=opt_r,
            device="cpu", gamma=0.99, algorithm="actor_critic",
        )

        metrics = trainer_r.train_epoch(num_episodes=10, epsilon=0.0)

        # Save RL checkpoint
        rl_path = tmp_path / "rl_best.pt"
        torch.save({
            "agent_state_dict": agent.state_dict(),
            "algorithm": "actor_critic",
            "config": {"backbone": "small", "hidden_dim": 256},
            "metrics": metrics,
        }, rl_path)

        assert rl_path.exists()
        assert ckpt_path.exists()
        assert metrics["accuracy"] >= 0


class TestCheckpointLoading:
    """Test CLI checkpoint loading logic."""

    def test_load_contrastive_checkpoint(self, tmp_path):
        """Test loading contrastive checkpoint."""
        from metapathpredict.cli import _load_model_from_checkpoint

        encoder = ContrastiveEncoder(in_channels=4, backbone="small")
        path = tmp_path / "contrastive.pt"
        torch.save({
            "encoder_state_dict": encoder.state_dict(),
            "config": {"backbone": "small"},
        }, path)

        result = _load_model_from_checkpoint(path, torch.device("cpu"))
        assert len(result) == 2
        model, model_type = result
        assert model_type == "contrastive"
        assert isinstance(model, ContrastiveEncoder)

    def test_load_rl_checkpoint(self, tmp_path):
        """Test loading RL agent checkpoint."""
        from metapathpredict.cli import _load_model_from_checkpoint

        agent = ActorCriticAgent(in_channels=4, num_actions=3, backbone="small")
        path = tmp_path / "rl.pt"
        torch.save({
            "agent_state_dict": agent.state_dict(),
            "algorithm": "actor_critic",
            "config": {"backbone": "small", "hidden_dim": 256},
        }, path)

        result = _load_model_from_checkpoint(path, torch.device("cpu"))
        assert len(result) == 3
        model, model_type, algorithm = result
        assert model_type == "rl"
        assert algorithm == "actor_critic"
        assert isinstance(model, ActorCriticAgent)

    def test_load_supervised_checkpoint(self, tmp_path):
        """A scripts/baselines.py checkpoint (ConfigurableCNN, key 'state_dict') loads as 'supervised'."""
        from metapathpredict.baselines import save_supervised_checkpoint
        from metapathpredict.cli import _load_model_from_checkpoint
        from metapathpredict.models.configurable_cnn import ConfigurableCNN

        model = ConfigurableCNN(num_classes=8, kernel_preset="small", base_channels=16)
        path = tmp_path / "supervised.pt"
        save_supervised_checkpoint(model, path, {
            "backbone": "small", "base_channels": 16, "norm": "batch",
            "num_classes": 8, "class_names": ["a", "b", "c", "d", "e", "f", "g", "h"],
        })

        result = _load_model_from_checkpoint(path, torch.device("cpu"))
        assert len(result) == 2
        loaded, model_type = result
        assert model_type == "supervised"
        assert isinstance(loaded, ConfigurableCNN)
        assert loaded.class_names == ["a", "b", "c", "d", "e", "f", "g", "h"]
        x = torch.zeros(1, 4, 100)
        with torch.no_grad():
            assert torch.allclose(loaded(x), model.eval()(x))

    def test_load_supervised_checkpoint_with_rc_share(self, tmp_path):
        """The RCShare wrapper (scripts/baselines.py --rc-share) round-trips too, not just the plain CNN."""
        from metapathpredict.baselines import build_classifier, save_supervised_checkpoint
        from metapathpredict.cli import _load_model_from_checkpoint
        from metapathpredict.models.configurable_cnn import RCShared

        model = build_classifier(3, "small", 16, "batch", "avg", "mean")
        path = tmp_path / "rcshare.pt"
        save_supervised_checkpoint(model, path, {
            "backbone": "small", "base_channels": 16, "norm": "batch", "pool": "avg",
            "rc_share": "mean", "num_classes": 3, "class_names": ["bacteria", "eukaryotic", "virus"],
        })

        loaded, model_type = _load_model_from_checkpoint(path, torch.device("cpu"))
        assert model_type == "supervised"
        assert isinstance(loaded, RCShared)

    def test_predict_single_dispatches_supervised_checkpoints(self, tmp_path):
        """_predict_single (used by evaluate/predict) runs a plain forward+softmax for 'supervised'."""
        import torch.nn.functional as F

        from metapathpredict.baselines import save_supervised_checkpoint
        from metapathpredict.cli import _load_model_from_checkpoint, _predict_single
        from metapathpredict.models.configurable_cnn import ConfigurableCNN

        model = ConfigurableCNN(num_classes=3, kernel_preset="small", base_channels=16).eval()
        path = tmp_path / "supervised.pt"
        save_supervised_checkpoint(model, path, {"backbone": "small", "base_channels": 16, "num_classes": 3})
        loaded, model_type = _load_model_from_checkpoint(path, torch.device("cpu"))

        x = torch.rand(1, 4, 100)
        class_idx, confidence, probs = _predict_single(loaded, model_type, x)
        expected = F.softmax(model(x), dim=1).squeeze(0)
        assert probs == pytest.approx(expected.tolist(), abs=1e-6)
        assert class_idx == int(expected.argmax())
        assert confidence == pytest.approx(expected.max().item(), abs=1e-6)

    def test_unrecognised_checkpoint_format_still_raises(self, tmp_path):
        from metapathpredict.cli import _load_model_from_checkpoint

        path = tmp_path / "junk.pt"
        torch.save({"something_else": {}}, path)
        with pytest.raises(ValueError):
            _load_model_from_checkpoint(path, torch.device("cpu"))


class TestPredictContigs:
    """predict_command slides a window across a sequence longer than the model's fragment size (Q-2/Q-3)."""

    def _checkpoint(self, tmp_path, num_classes=3):
        from metapathpredict.baselines import save_supervised_checkpoint
        from metapathpredict.models.configurable_cnn import ConfigurableCNN

        model = ConfigurableCNN(num_classes=num_classes, kernel_preset="small", base_channels=16).eval()
        path = tmp_path / "model.pt"
        save_supervised_checkpoint(model, path, {
            "backbone": "small", "base_channels": 16, "num_classes": num_classes,
            "class_names": ["bacteria", "eukaryotic", "virus"][:num_classes],
        })
        return path

    def _fasta(self, tmp_path, records):
        path = tmp_path / "in.fasta"
        with open(path, "w") as f:
            for seq_id, seq in records:
                f.write(f">{seq_id}\n{seq}\n")
        return path

    def test_a_short_sequence_uses_one_window_a_long_one_slides(self, tmp_path):
        import random

        from metapathpredict.cli import predict_command

        random.seed(0)
        model_path = self._checkpoint(tmp_path)
        short = "".join(random.choice("ACGT") for _ in range(100))
        long_contig = "".join(random.choice("ACGT") for _ in range(1350))  # 2 full 500bp windows + a dropped tail
        fasta = self._fasta(tmp_path, [("short_seq", short), ("long_contig", long_contig)])
        out = tmp_path / "out.tsv"

        config_path = tmp_path / "predict.yaml"
        config_path.write_text("data:\n  default_fragment_size: 500\n")
        args = argparse.Namespace(model=str(model_path), input=str(fasta), output=str(out), config=str(config_path),
                                  device="cpu", window_step=None)
        assert predict_command(args) == 0

        rows = {r["sequence_id"]: r for r in csv.DictReader(open(out), delimiter="\t")}
        assert rows["short_seq"]["num_windows"] == "1" and rows["short_seq"]["length"] == "100"
        assert rows["long_contig"]["num_windows"] == "2" and rows["long_contig"]["length"] == "1350"
        for row in rows.values():
            assert row["predicted_class"] in ("bacteria", "eukaryotic", "virus")
            probs = [float(row[f"prob_{c}"]) for c in ("bacteria", "eukaryotic", "virus")]
            assert pytest.approx(sum(probs), abs=1e-3) == 1.0

    def test_window_step_controls_overlap_of_the_sliding_windows(self, tmp_path, monkeypatch):
        from metapathpredict.cli import predict_command

        model_path = self._checkpoint(tmp_path)
        fasta = self._fasta(tmp_path, [("c", "A" * 1500)])
        out = tmp_path / "out.tsv"

        calls = []
        import metapathpredict.contigs as contigs_module
        original = contigs_module.classify_contig

        def spy(probs_fn, sequence, window, step=None):
            calls.append(step or window)
            return original(probs_fn, sequence, window, step)

        monkeypatch.setattr("metapathpredict.contigs.classify_contig", spy)
        config_path = tmp_path / "predict.yaml"
        config_path.write_text("data:\n  default_fragment_size: 500\n")
        args = argparse.Namespace(model=str(model_path), input=str(fasta), output=str(out), config=str(config_path),
                                  device="cpu", window_step=250)
        assert predict_command(args) == 0
        assert calls == [250]
        rows = list(csv.DictReader(open(out), delimiter="\t"))
        assert rows[0]["num_windows"] == str(len(range(0, 1500 - 500 + 1, 250)))
