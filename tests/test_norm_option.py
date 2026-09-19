"""GroupNorm/LayerNorm option: no statistics shared between samples, and it survives a checkpoint round trip."""

import pytest
import torch
import torch.nn as nn

from metapathpredict.cli import _load_model_from_checkpoint
from metapathpredict.config.settings import ContrastiveConfig
from metapathpredict.models.base import group_count
from metapathpredict.models.contrastive import ContrastiveEncoder
from metapathpredict.models.reinforcement import ActorCriticAgent
from metapathpredict.probe import recalibrate_batchnorm

BATCH_NORMS = nn.modules.batchnorm._BatchNorm


def _encoder(norm: str) -> ContrastiveEncoder:
    return ContrastiveEncoder(
        backbone="small", projection_dim=32, hidden_dim=64, base_channels=16, num_classes=3, norm=norm
    )


def _x(n: int = 8, length: int = 64) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.randint(0, 4, (n, length)), 4).permute(0, 2, 1).float()


def test_group_count_divides_channels():
    assert [group_count(c) for c in (16, 32, 48, 128, 512)] == [16, 32, 24, 32, 32]


def test_group_norm_model_has_no_batchnorm_and_default_keeps_it():
    assert not any(isinstance(m, BATCH_NORMS) for m in _encoder("group").modules())
    assert any(isinstance(m, BATCH_NORMS) for m in _encoder("batch").modules())


def test_train_mode_output_of_a_sample_does_not_depend_on_the_rest_of_the_batch():
    x = _x()
    other_batch = torch.cat([x[:1], _x(7)])  # same first sample, different companions
    for norm, independent in (("group", True), ("batch", False)):
        torch.manual_seed(0)
        model = _encoder(norm).train()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        same = torch.allclose(model(x)[:1], model(other_batch)[:1], atol=1e-5)
        assert same == independent, norm


def test_invalid_norm_is_rejected():
    with pytest.raises(ValueError):
        _encoder("layer")
    with pytest.raises(ValueError):
        ContrastiveConfig(norm="layer")


def test_recalibration_is_a_no_op_without_batchnorm():
    model = _encoder("group")
    loader = [(_x(), torch.zeros(8, dtype=torch.long))]
    assert recalibrate_batchnorm(model.encoder, loader, "cpu") == 0
    assert not model.encoder.training


def test_contrastive_checkpoint_round_trip_keeps_group_norm(tmp_path):
    model = _encoder("group")
    cfg = ContrastiveConfig(backbone="small", base_channels=16, projection_dim=32, hidden_dim=64, norm="group")
    path = tmp_path / "c.pt"
    torch.save({"encoder_state_dict": model.state_dict(), "config": cfg.model_dump(), "num_classes": 3}, path)
    loaded, kind = _load_model_from_checkpoint(path, torch.device("cpu"))
    assert kind == "contrastive"
    assert not any(isinstance(m, BATCH_NORMS) for m in loaded.modules())
    x = _x()
    assert torch.allclose(loaded(x), model.eval()(x), atol=1e-5)


def test_rl_agent_accepts_norm_and_saves_it_in_the_checkpoint(tmp_path):
    agent = ActorCriticAgent(num_actions=3, backbone="small", hidden_dim=64, base_channels=16, norm="group")
    assert not any(isinstance(m, BATCH_NORMS) for m in agent.encoder.modules())
    path = tmp_path / "rl.pt"
    torch.save({"agent_state_dict": agent.state_dict(), "config": {"backbone": "small", "hidden_dim": 64},
                "algorithm": "actor_critic", "base_channels": 16, "norm": "group", "num_classes": 3}, path)
    loaded, kind, _ = _load_model_from_checkpoint(path, torch.device("cpu"))
    assert kind == "rl"
    assert not any(isinstance(m, BATCH_NORMS) for m in loaded.encoder.modules())
