import numpy as np
import pytest
import torch

from balr.losses.sparsity_loss import SparsityLoss


def test_sparsity_loss_raises_exception():
    encoder_dim = 4
    M_samples_per_class = 2
    lf = SparsityLoss()

    Z = torch.Tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5, 0.6],
            [0.5, 0.6, 0.7, 0.8],
            [0.7, 0.8, 0.9, 1.0],
            [0.9, 1.0, 0.1, 0.2],
        ]
    )
    labels = torch.Tensor([0, 0, 0, 1, 1])

    with pytest.raises(
        RuntimeError, match="SparsityLoss function has not been properly setup."
    ):
        lf(
            None,
            labels,
            None,
            None,
            Z,
        )

    lf.setup(M_samples_per_class=M_samples_per_class, internal_dim=encoder_dim)
    with pytest.raises(ValueError, match="Error computing sparsity loss."):
        lf(
            None,
            labels,
            None,
            None,
            Z,
        )


def test_sparsity_loss():
    encoder_dim = 4
    M_samples_per_class = 2
    lf = SparsityLoss()
    lf.setup(M_samples_per_class=M_samples_per_class, internal_dim=encoder_dim)

    Z = torch.Tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5, 0.6],
            [0.5, 0.6, 0.7, 0.8],
            [0.7, 0.8, 0.9, 1.0],
            [0.9, 1.0, 0.1, 0.2],
            [0.2, 0.3, 0.4, 0.5],
        ]
    )
    labels = torch.Tensor([0, 0, 1, 1, 2, 2])

    loss = lf(
        None,
        labels,
        None,
        None,
        Z,
    )

    expected_activations = [Z[0:2].sum(0), Z[2:4].sum(0), Z[4:].sum(0)]
    expected_activations = np.stack(expected_activations)
    expected_loss = (
        torch.from_numpy(expected_activations - lf.target).clamp(0, 1).pow(2).sum()
    ) / len(expected_activations)
    assert torch.equal(loss, expected_loss)
