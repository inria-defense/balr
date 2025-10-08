from unittest.mock import patch

import pytest
import torch

from balr.scoring.beta import DirichletMultinomialScorer


def test_scorers_raise_errors_when_invalid_referential_shape():
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


@patch("balr.scoring.scoring.LOGGER")
def test_scorers_logs_warning_when_not_fitted(mock_logger):
    scorer = DirichletMultinomialScorer(K=2, n_attributes=3)

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

    scorer(enroll, test)
    mock_logger.error.assert_called_once()

    scorer.loaded = True
    mock_logger.error.reset_mock()
    scorer(enroll, test)
    mock_logger.error.assert_not_called()
