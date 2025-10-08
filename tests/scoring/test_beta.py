import pytest
import torch

from balr.scoring.beta import DirichletMultinomialScorer


def test_beta_scorer_raises_errors_when_invalid_referential_shape():
    scorer = DirichletMultinomialScorer(K=5, n_attributes=256)

    with pytest.raises(
        ValueError, match="Referential tensor should be of shape N x n x K"
    ):
        scorer.fit(torch.zeros((2, 128)), 0)

    with pytest.raises(
        ValueError, match="Referential tensor should be of shape N x n x K with n=256"
    ):
        scorer.fit(torch.zeros((2, 128, 4)), 0)

    with pytest.raises(
        ValueError, match="Referential tensor should be of shape N x n x K with K=5"
    ):
        scorer.fit(torch.zeros((2, 256, 4)), 0)


def test_beta_scorer_approximate_parameter_values(ground_referential: torch.Tensor):
    scorer = DirichletMultinomialScorer(K=2, n_attributes=1)
    scorer.fit(ground_referential, 500)

    assert scorer.alpha.shape == (1, 2)
    assert torch.allclose(scorer.alpha, torch.tensor([[0.5, 0.2]]), rtol=0.1)


def test_beta_scorer_forward():
    scorer = DirichletMultinomialScorer(K=2, n_attributes=3)
    scorer.alpha = torch.tensor([[0.5, 0.2], [0.1, 0.8], [0.3, 0.4]])

    enroll = torch.tensor(
        [
            [[0, 2], [1, 1], [2, 0]],
            [[2, 1], [1, 2], [3, 0]],
            [[0, 2], [1, 1], [2, 0]],
            [[5, 0], [5, 0], [5, 0]],
        ]
    )

    test = torch.tensor(
        [
            [[0, 2], [1, 1], [2, 0]],
            [[5, 0], [5, 0], [5, 0]],
            [[2, 1], [1, 2], [3, 0]],
            [[2, 1], [1, 2], [3, 0]],
        ]
    )

    llrs = scorer(enroll, test)
    assert llrs.shape == (4,)

    expected_llrs = torch.tensor([3.4115, 0.1956, 1.8651, 0.1956])
    assert torch.allclose(llrs, expected_llrs, rtol=0.001)
