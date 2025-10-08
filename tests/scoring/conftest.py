import numpy as np
import numpy.typing as npt
import pytest
import torch
from scipy import stats

NB_ATTRIBUTES = 128
NB_SPEAKERS = 500


def binomial_speaker_statistics(
    n_attributes: int = NB_ATTRIBUTES,
    n_speakers: int = NB_SPEAKERS,
    f0: float = 0.3,
    p0: float = 0.8,
    q0: float = 0.3,
) -> torch.Tensor:
    """
    Generate speaker binary attribute activation statistics using a distribution
    modeled by f (probability that a speaker possesses an attribute), p the activation
    frequency of the attribute for speakers who possess the attribute and q the
    activation frequency of the attribute for speakers who do not possess the attribute.

    Args:
        n_attributes (int, optional): number of attributes. Defaults to NB_ATTRIBUTES.
        n_speakers (int, optional): number of speakers. Defaults to NB_SPEAKERS.
        f0 (float, optional): ground truth f parameter. Defaults to 0.3.
        p0 (float, optional): ground truth p parameter. Defaults to 0.8.
        q0 (float, optional): ground truth q parameter. Defaults to 0.3.

    Returns:
        torch.Tensor: the activation statistics
    """
    activations: list[torch.Tensor] = []
    non_activations: list[torch.Tensor] = []

    for _ in range(n_attributes):
        speaker_attributes = np.random.binomial(n=1, p=f0, size=n_speakers)
        N = torch.from_numpy(np.random.randint(20, 200, n_speakers))
        samples = torch.tensor(
            [
                (
                    np.random.binomial(N[i], p0)
                    if selection == 1
                    else np.random.binomial(N[i], q0)
                )
                for i, selection in enumerate(speaker_attributes)
            ]
        )
        activations.append(samples)
        non_activations.append(N - samples)

    return torch.stack(
        (torch.stack(activations, dim=-1), torch.stack(non_activations, dim=-1)),
        dim=-1,
    )


def speaker_statistics(
    n_attributes: int = NB_ATTRIBUTES,
    n_speakers: int = NB_SPEAKERS,
    alpha_gt: npt.NDArray | None = None,
) -> torch.Tensor:
    """
    Generate speaker binary attribute activation statistics using a
    beta distribution.

    Args:
        n_attributes (int, optional): number of attributes. Defaults to NB_ATTRIBUTES.
        n_speakers (int, optional): number of speakers. Defaults to NB_SPEAKERS.
        alpha_gt (npt.NDArray | None, optional): ground truth alpha parameters for the
            beta distribution. If None, random values are chosen for each attribute.
            Defaults to None.

    Returns:
        torch.Tensor: the fixture speaker statistics
    """
    activations = []
    non_activations = []
    for _ in range(n_attributes):
        if alpha_gt is None:
            alpha = np.random.rand(2)
        else:
            alpha = alpha_gt
        p = stats.dirichlet.rvs(alpha, n_speakers)
        N = np.random.randint(20, 200, len(p))
        n = torch.from_numpy(
            np.vstack([stats.multinomial.rvs(N_, p_) for N_, p_ in zip(N, p)])
        )
        activations.append(n[:, 0])
        non_activations.append(n[:, 1])

    return torch.stack(
        (torch.stack(activations, dim=-1), torch.stack(non_activations, dim=-1)), dim=-1
    )


@pytest.fixture(scope="package")
def referential() -> torch.Tensor:
    """
    Generate attribute activations for NB_ATTRIBUTES with random alpha parameters
    for the dirichlet distribution.

    Returns:
        torch.Tensor: tensor of dimension NB_SPEAKERS x NB_ATTRIBUTES x 2
    """
    speaker_stats = speaker_statistics()
    return speaker_stats


@pytest.fixture(scope="package")
def ground_referential() -> torch.Tensor:
    """
    Generate attribute activations with fixed ground truth alpha parameters
    and only 1 attribute.

    Returns:
        torch.Tensor: tensor of dimension NB_SPEAKERS x 1 x 2
    """
    speaker_stats = speaker_statistics(n_attributes=1, alpha_gt=np.array([0.5, 0.2]))
    return speaker_stats


@pytest.fixture(scope="package")
def binomial_ground_referential() -> torch.Tensor:
    """
    Generate attribute activations with fixed ground truth alpha parameters
    and only 1 attribute.

    Returns:
        torch.Tensor: tensor of dimension NB_SPEAKERS x 1 x 2
    """
    speaker_stats = binomial_speaker_statistics(n_attributes=1, f0=0.3, p0=0.8, q0=0.3)
    return speaker_stats
