import logging
from pathlib import Path

import torch

from balr.scoring.scoring import Scorer, check_tensor_shape

LOGGER = logging.getLogger(__name__)


class DirichletMultinomialScorer(Scorer):
    """
    The DirichletMultinomialScorer models a dirichlet multinomial distribution,
    with n attributes and K values per attribute.

    For binary attributes, with K = 2, the Dirichlet-Multinomial model is a
    Beta-Bernouilli model where the probability distribution of activating an
    attribute for a speaker is parameterized by a Beta distribution with parameters
    α and β.

    The Beta-Bernoulli scoring is a special case of a Dirichlet-Multinomial model
    (with K=2 parameters).
    """

    def __init__(
        self,
        K: int = 2,
        eps: float = 1e-8,
        n_attributes: int = 512,
        checkpoint_path: str | Path | None = None,
        save_dir: str | Path | None = None,
        exist_ok: bool = False,
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize a DirichletMultinomialScorer.

        Args:
            K (int, optional): number of quantized states per attribute
                (2 for binary attributes). Defaults to 2.
            eps (float, optional): non zero small value for numerical stability.
                Defaults to 1e-8.
            n_attributes (int, optional): number of attributes. Defaults to 512.
            checkpoint_path (str | Path | None, optional): checkpoint path to load the
                model weights from. Defaults to None.
            save_dir (str | Path | None, optional): directory where the model weights
                will be saved. Defaults to None.
            exist_ok (bool, optional): Save results in save_dir even if it already exists.
                If false, save_dir will be incremented with a suffix if it already exists.
                Defaults to False.
            device (str | torch.device, optional): device for computation.
                Defaults to "cpu".
        """
        super().__init__(n_attributes, checkpoint_path, save_dir, exist_ok, device)
        self.eps = eps
        self.K = K
        self.register_buffer("alpha", torch.zeros((n_attributes, K)))
        self.load_model()

    def _score(
        self, enrollment_activations: torch.Tensor, test_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute score values for all pairs of attributes.

        Args:
            enrollment_activations (torch.Tensor): enrollment attribute activations of
                dimension N x n x K where N is the number of enrollment elements,
                n is the number of attributes, and K is the number of possible values
                for each attribute (i.e. 2 in case of binary attributes).
            test_activations (torch.Tensor): test attribute activations of dimension
                N x n x K

        Returns:
            torch.Tensor: score values of dimension N.
        """
        return (
            torch.lgamma((test_activations + self.alpha).sum(-1))
            + torch.lgamma((enrollment_activations + self.alpha).sum(-1))
            - torch.lgamma(
                (test_activations + enrollment_activations + self.alpha).sum(-1)
            )
            - torch.lgamma(self.alpha.sum(-1))
            + torch.lgamma(test_activations + enrollment_activations + self.alpha).sum(-1)
            + torch.lgamma(self.alpha).sum(-1)
            - torch.lgamma(enrollment_activations + self.alpha).sum(-1)
            - torch.lgamma(test_activations + self.alpha).sum(-1)
        ).sum(-1)

    def _fit_prior(
        self, referential: torch.Tensor, n_iterations: int, *args, **kwargs
    ) -> None:
        """
        Estimate scorer parameters from the activation statistics of a referential.

        Args:
            referential (torch.Tensor): a tensor of activation statistics of dimension
                N x n x K where N is the number of elements of the referential,
                n is the number of attributes, and K is the number of possible values
                for each attribute (i.e. 2 in case of binary attributes).
            n_iterations (int): number of iterations for parameter estimations.
        """
        check_tensor_shape(referential, self.n_attributes, self.K, "Referential")

        LOGGER.info(
            "Estimating parameters with Expectation-Maximization "
            f"({n_iterations} iterations)."
        )

        p_ = referential / (referential.sum(dim=-1, keepdim=True) + self.eps)
        Ep = p_.mean(dim=0)
        gmean = (Ep * (1 - Ep) + 2 * self.eps) / (p_.var(dim=0) + self.eps) - 1.0
        gmean = gmean.log().mean().exp()
        alpha = Ep * gmean + self.eps

        for _ in range(n_iterations):
            sum_alpha = alpha.sum(dim=1)
            num = alpha * (torch.digamma(referential + alpha) - torch.digamma(alpha)).sum(
                dim=0
            )
            denom = (
                torch.digamma(referential.sum(dim=-1) + sum_alpha)
                - torch.digamma(sum_alpha)
            ).sum(dim=0)
            alpha = num / denom.unsqueeze(1)

        self.alpha = alpha

    def __str__(self):
        repr_str = (
            f"{self.__class__.__name__}: K={self.K}, device={self.device}.\n"
            f"alpha: shape={self.alpha.shape}"
        )
        return repr_str
