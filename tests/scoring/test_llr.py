import math

import pytest
import torch

from balr.scoring.llr import DNALLRScorer, MaxLLRScorer, SpeechLLRScorer
from tests.scoring.conftest import NB_ATTRIBUTES


def test_llr_scorer_computes_typicality_and_dropout(referential: torch.Tensor):
    scorer = DNALLRScorer(n_attributes=NB_ATTRIBUTES)
    scorer.fit(referential, 0)

    assert scorer.typicality.shape == (NB_ATTRIBUTES,)
    assert scorer.dropout.shape == (NB_ATTRIBUTES,)


def test_llr_scorer_typicality_and_dropout_values(ground_referential: torch.Tensor):
    scorer = SpeechLLRScorer(n_attributes=1)
    scorer.fit(ground_referential, 0)

    # typicality value should be close to 0.94
    assert math.isclose(scorer.typicality[0].item(), 0.94, abs_tol=0.06)
    # dropout value should be close to 0.26
    assert math.isclose(scorer.dropout[0].item(), 0.26, abs_tol=0.06)


def test_approximate_llrs_for_speech_scorer(ground_referential: torch.Tensor):
    scorer = SpeechLLRScorer(n_attributes=1, drop_in=0.28)
    scorer.fit(ground_referential, 0)

    LR_00 = scorer.llrs[0, :, 0]
    assert torch.allclose(LR_00, torch.tensor([0.16]), atol=0.15)
    LR_11 = scorer.llrs[1, :, 1]
    assert torch.allclose(LR_11, torch.tensor([0.1316]), atol=0.15)
    LR_01 = scorer.llrs[0, :, 1]
    LR_10 = scorer.llrs[1, :, 0]
    assert torch.allclose(LR_01, torch.tensor([-1.27]), atol=0.15)
    assert torch.equal(LR_01, LR_10)


def test_approximate_llrs_for_dna_scorer(ground_referential: torch.Tensor):
    scorer = DNALLRScorer(n_attributes=1, drop_in=0.28)
    scorer.fit(ground_referential, 0)

    LR_00 = scorer.llrs[0, :, 0]
    assert torch.allclose(LR_00, torch.tensor([0.16]), atol=0.15)
    LR_11 = scorer.llrs[1, :, 1]
    assert torch.allclose(LR_11, torch.tensor([0.1316]), atol=0.15)
    LR_01 = scorer.llrs[0, :, 1]
    LR_10 = scorer.llrs[1, :, 0]
    assert torch.allclose(LR_01, torch.tensor([-1.27]), atol=0.15)
    assert torch.equal(LR_01, LR_10)


def test_llr_scorer_raises_error_when_more_than_one_activation():
    scorer = DNALLRScorer(n_attributes=2)

    enroll = torch.tensor([[[2, 0], [0, 2]], [[1, 1], [2, 0]]])
    test = torch.tensor([[[1, 1], [0, 2]], [[2, 0], [0, 2]]])

    with pytest.raises(
        ValueError,
        match="LLRScorer does not support computing LLR scores for more than 1",
    ):
        scorer(enroll, test)


def test_llr_forward():
    n_attributes = 128
    N_samples = 500
    scorer = DNALLRScorer(n_attributes=n_attributes)
    scorer.typicality = torch.rand(n_attributes)
    scorer.dropout = torch.rand(n_attributes)
    scorer._precompute_llrs()

    activations = torch.randint(0, 2, (N_samples, n_attributes))
    enroll = torch.stack([activations, 1 - activations], dim=2)

    activations = torch.randint(0, 2, (N_samples, n_attributes))
    test = torch.stack([activations, 1 - activations], dim=2)

    llrs = scorer(enroll, test)
    assert llrs.shape == (N_samples,)

    def llr(enroll: torch.Tensor, test: torch.Tensor) -> float:
        return sum(
            [
                scorer.llrs[1 - enroll[i, 0], i, 1 - test[i, 0]].item()
                for i in range(n_attributes)
            ]
        )

    expected_llrs = torch.tensor(
        [llr(enroll[i, :, :], test[i, :, :]) for i in range(N_samples)]
    )
    assert torch.allclose(llrs, expected_llrs)


def test_maxllr_scorer_computes_parameters(referential: torch.Tensor):
    scorer = MaxLLRScorer(n_attributes=NB_ATTRIBUTES)
    scorer.fit(referential, 10)

    assert scorer.f.shape == (NB_ATTRIBUTES,)
    assert scorer.p.shape == (NB_ATTRIBUTES,)
    assert scorer.q.shape == (NB_ATTRIBUTES,)


def test_maxllr_approximate_parameter_values(binomial_ground_referential: torch.Tensor):
    scorer = MaxLLRScorer(n_attributes=1)
    scorer.fit(binomial_ground_referential, 100)

    # f value should approximate f0 (0.3)
    assert torch.allclose(scorer.f, torch.tensor([0.3], dtype=torch.float64), atol=0.05)
    # p value should approximate p0 (0.8)
    assert torch.allclose(scorer.p, torch.tensor([0.8], dtype=torch.float64), atol=0.05)
    # q value should approximate q0 (0.3)
    assert torch.allclose(scorer.q, torch.tensor([0.3], dtype=torch.float64), atol=0.05)


def test_maxllr_forward():
    scorer = MaxLLRScorer(n_attributes=2)
    scorer.f = torch.tensor([0.3, 0.4])
    scorer.p = torch.tensor([0.8, 0.9])
    scorer.q = torch.tensor([0.3, 0.1])

    enroll = torch.tensor([[[2, 0], [0, 2]], [[1, 1], [2, 0]]])
    test = torch.tensor([[[1, 1], [0, 2]], [[2, 0], [0, 2]]])

    llrs = scorer(enroll, test)
    assert llrs.shape == (2,)

    def likelihood(f: float, p: float, q: float, acts: float, nonacts: float) -> float:
        return f * (p**acts) * ((1 - p) ** nonacts) + (1 - f) * (q**acts) * (
            (1 - q) ** nonacts
        )

    def llr(enroll: torch.Tensor, test: torch.Tensor) -> float:
        res = 0.0
        for attribute in range(2):
            f = scorer.f[attribute].item()
            p = scorer.p[attribute].item()
            q = scorer.q[attribute].item()

            ae = enroll[attribute, 0].item()
            ne = enroll[attribute, 1].item()

            at = test[attribute, 0].item()
            nt = test[attribute, 1].item()

            res += (
                math.log(likelihood(f, p, q, ae + at, ne + nt))
                - math.log(likelihood(f, p, q, ae, ne))
                - math.log(likelihood(f, p, q, at, nt))
            )

        return res

    expected_llrs = torch.tensor([llr(enroll[i, :, :], test[i, :, :]) for i in range(2)])
    assert torch.allclose(llrs, expected_llrs)
