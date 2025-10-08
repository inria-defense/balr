from unittest.mock import patch

import numpy as np
import pytest
import torch

from balr.samplers.nxm_samplers import ExhaustiveNxMSampler, RandomNxMSampler


@pytest.fixture()
def patch_random_gen():
    # patch numpy random generator with seed for reproducible results
    seeded_rng = np.random.default_rng(seed=1234)
    with patch("balr.samplers.nxm_samplers.np.random.default_rng") as mocked_rng:
        mocked_rng.return_value = seeded_rng
        yield


def test_nmsampler_raises_error_if_batch_size_invalid():
    labels = torch.IntTensor(range(12))
    with pytest.raises(ValueError, match="Invalid batch_size"):
        RandomNxMSampler(labels, 3, 3, 10)


def test_random_sampler(patch_random_gen):
    labels = torch.IntTensor([0, 0, 1, 1, 2, 2])
    sampler = RandomNxMSampler(labels, 2, 1, shuffle=False, drop_last=True)
    samples = list(sampler)
    assert samples == [1, 3]

    sampler = RandomNxMSampler(labels, 2, 2, shuffle=False, drop_last=True)
    samples = list(sampler)
    assert set(samples) == {0, 1, 2, 3}

    sampler = RandomNxMSampler(labels, 2, 1, shuffle=False, drop_last=False)
    samples = list(sampler)
    assert samples == [0, 2, 4, 0]

    sampler = RandomNxMSampler(labels, 2, 2, shuffle=False, drop_last=False)
    samples = list(sampler)
    assert len(samples) == 8
    assert set(samples[:4]) == {0, 1, 2, 3}
    assert set(samples[4:]) == {4, 5, 0, 1}

    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
    sampler = RandomNxMSampler(labels, 2, 2, shuffle=False, drop_last=True)
    assert list(sampler) == [3, 1, 5, 7]

    sampler = RandomNxMSampler(labels, 2, 2, shuffle=False, drop_last=False)
    assert list(sampler) == [3, 4, 6, 7, 8, 9, 0, 3]


def test_exhaustive_sampler(patch_random_gen):
    labels = torch.tensor([0, 1, 2, 0, 2, 0])
    sampler = ExhaustiveNxMSampler(labels, 2, 2, shuffle=False)
    assert list(sampler) == [0, 3, 1, 1, 2, 4, 5, 0]

    sampler = ExhaustiveNxMSampler(labels, 2, 2, shuffle=True)
    assert list(sampler) == [0, 5, 1, 1, 4, 2, 0, 3]

    labels = torch.tensor([0, 1, 2, 0, 2, 0, 2])
    sampler = ExhaustiveNxMSampler(labels, 2, 2, shuffle=False)
    assert list(sampler) == [0, 3, 1, 1, 2, 4, 5, 0]

    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
    sampler = ExhaustiveNxMSampler(labels, 2, 2, shuffle=False)
    assert list(sampler) == [0, 1, 5, 6, 8, 9, 2, 3, 7, 5, 4, 0]


def test_distributed_exhaustive_sampler(patch_random_gen):
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
    sampler = ExhaustiveNxMSampler(labels, 2, 2, shuffle=False, num_replicas=2, rank=0)
    assert list(sampler) == [0, 1, 5, 6, 7, 5, 4, 0]
    sampler = ExhaustiveNxMSampler(
        labels, 2, 2, shuffle=False, drop_last=True, num_replicas=2, rank=0
    )
    assert list(sampler) == [0, 1, 5, 6]

    sampler = ExhaustiveNxMSampler(labels, 2, 2, shuffle=False, num_replicas=2, rank=1)
    assert list(sampler) == [8, 9, 2, 3, 0, 1, 5, 6]
    sampler = ExhaustiveNxMSampler(
        labels, 2, 2, shuffle=False, drop_last=True, num_replicas=2, rank=1
    )
    assert list(sampler) == [8, 9, 2, 3]


def test_distributed_random_sampler():
    # test the sampler uses deterministic shuffles until the epoch changes
    # when using distributed sampling
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
    sampler = RandomNxMSampler(labels, 2, 2, shuffle=True, num_replicas=1, rank=0)
    assert list(sampler) == [1, 3, 5, 6, 9, 8, 2, 0]
    assert list(sampler) == [1, 3, 5, 6, 9, 8, 2, 0]

    sampler.set_epoch(1)
    assert list(sampler) == [4, 0, 5, 6, 8, 9, 1, 0]

    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    sampler = RandomNxMSampler(labels, 2, 2, shuffle=True, num_replicas=2, rank=0)
    assert list(sampler) == [4, 0, 15, 16]
    sampler = RandomNxMSampler(labels, 2, 2, shuffle=True, num_replicas=2, rank=1)
    assert list(sampler) == [7, 5, 13, 10]
