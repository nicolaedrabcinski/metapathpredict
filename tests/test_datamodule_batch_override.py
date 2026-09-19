"""SequenceDataModule loaders accept a per-phase batch size override."""

import torch
from torch.utils.data import TensorDataset

from metapathpredict.data import SequenceDataModule


def _module(batch_size=16):
    ds = TensorDataset(torch.zeros(100, 4, 10), torch.zeros(100, dtype=torch.long))
    return SequenceDataModule(
        train_dataset=ds, val_dataset=ds, test_dataset=ds, batch_size=batch_size, num_workers=0,
    )


def test_default_train_loader_is_cached_and_uses_module_batch_size():
    dm = _module(16)
    assert dm.train_dataloader() is dm.train_dataloader()
    assert dm.train_dataloader().batch_size == 16


def test_train_override_gives_separate_loader_with_uniform_batches():
    dm = _module(16)
    loader = dm.train_dataloader(batch_size=32)
    assert loader is not dm.train_dataloader()
    assert loader.batch_size == 32 and loader.drop_last
    assert all(len(x) == 32 for x, _ in loader)  # 100 samples -> 3 full batches, remainder dropped


def test_same_batch_size_as_module_reuses_cached_loader():
    dm = _module(16)
    assert dm.train_dataloader(batch_size=16) is dm.train_dataloader()


def test_val_override_keeps_batches_uniform():
    dm = _module(16)
    loader = dm.val_dataloader(batch_size=32)
    assert loader.batch_size == 32 and loader.drop_last and not isinstance(loader.sampler, torch.utils.data.RandomSampler)
