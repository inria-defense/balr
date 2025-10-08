import logging
from abc import abstractmethod
from itertools import product
from math import log
from pathlib import Path

import torch
from torch import Tensor

from balr.scoring.scoring import Scorer, check_tensor_shape

LOGGER = logging.getLogger(__name__)


class LLRScorer(Scorer):
    """
    Reference LLR scorer from Imen Ben Amor's thesis that models the distribution
    of attributes over a reference population with three parameters: typicality,
    drop-out and drop-in.

    The typicality $T_i$ of a binary attribute $b_i$ is the frequency of speaker pairs
    sharing the attribute in the reference population.

    The drop-out $Dout_i$ of attribute $b_i$ is defined as the probability of the
    attribute disappearing from the profile.

    The drop-in is the probability of encountering noise leading to a false detection
    of the attribute in a recording.

    In this scoring model, each speaker either possesses or does not possess each
    attribute.
    """

    def __init__(
        self,
        n_attributes: int = 512,
        drop_in: float = 0.12,
        typicality_threshold: float = 1e-4,
        checkpoint_path: str | Path | None = None,
        save_dir: str | Path | None = None,
        exist_ok: bool = False,
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(n_attributes, checkpoint_path, save_dir, exist_ok, device)
        self.drop_in = drop_in
        self.typicality_threshold = typicality_threshold
        self.register_buffer("typicality", torch.zeros(n_attributes))
        self.register_buffer("dropout", torch.zeros(n_attributes))
        self.register_buffer("llrs", torch.zeros((2, n_attributes, 2)))
        self.load_model()

    def _fit_prior(self, referential: Tensor, n_iterations: int, *args, **kwargs) -> None:
        """
        Estimate typicality and dropout values for each binary attribute from the
        activation statistics of a referential and precompute a matrix of LLR values
        for all possible combinations of enrollment and test activation values.

        Args:
            referential (Tensor): a tensor of activation statistics of dimension
                N x n x K where N is the number of elements of the referential,
                n is the number of attributes, and K is the number of possible values
                for each attribute (i.e. 2 in case of binary attributes).
            n_iterations (int): number of iterations for parameter estimations.
        """
        check_tensor_shape(referential, self.n_attributes, 2, "Referential")

        LOGGER.info("Estimating typicality and dropout parameters.")
        proportions = referential[:, :, 0] / referential.sum(dim=-1)
        profile = referential[:, :, 0] > 0
        n_speakers = referential.shape[0]
        n_activations = profile.sum(dim=0)
        self.typicality = (n_activations * (n_activations - 1)) / (
            n_speakers * (n_speakers - 1)
        )
        self.dropout = 1 - (proportions.sum(dim=0) / profile.sum(dim=0).clamp(min=1.0))

        LOGGER.info("Precomputing LLR values for possible binary attribute activations.")
        self._precompute_llrs()

    @abstractmethod
    def LR_00(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def LR_10(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def LR_01(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def LR_11(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        pass

    def LLR(
        self, attribute: int, enroll_activations: int, test_activations: int
    ) -> float:
        """
        LLR computation for a single attribute. Non zero activation statistics are
        considered as ones.

        Args:
            attribute (int): attribute index
            enroll_activations (int): Number of activations of the attribute for
                enrollment
            test_activations (int): Number of activations of the attribute for test

        Returns:
            float: LLR value for the attribute
        """
        enroll_activations = min(enroll_activations, 1)
        test_activations = min(test_activations, 1)
        typicality = self.typicality[attribute]
        drop_out = self.dropout[attribute]

        if typicality < self.typicality_threshold:
            return log(1.0)

        # possible combinations of enrollment and test activation values
        # are (0, 0), (0, 1), (1, 0) and (1, 1)
        if enroll_activations == 0 and test_activations == 0:
            return log(self.LR_00(typicality, drop_out))
        if enroll_activations == 1 and test_activations == 1:
            return log(self.LR_11(typicality, drop_out))

        return log(
            (self.LR_01(typicality, drop_out) + self.LR_10(typicality, drop_out)) / 2
        )

    def _precompute_llrs(self) -> None:
        llrs = torch.zeros((2, self.n_attributes, 2), device=self.device)
        for i in range(self.n_attributes):
            for e, t in product([0, 1], repeat=2):
                llrs[e, i, t] = self.LLR(i, 1 - e, 1 - t)
        self.llrs = llrs

    def _score(self, enrollment_activations: Tensor, test_activations: Tensor) -> Tensor:
        check_tensor_shape(enrollment_activations, self.n_attributes, 2, "Enrollment")
        check_tensor_shape(test_activations, self.n_attributes, 2, "Test")

        if torch.any(enrollment_activations > 1) or torch.any(test_activations > 1):
            raise ValueError(
                "LLRScorer does not support computing LLR scores for more than 1 "
                "enrollment or test samples. The sum of activations and non activations "
                "must be 1 for each attribute of the enrollment and test embeddings."
            )

        enrollment_activations = enrollment_activations.float()
        test_activations = test_activations.float()

        enroll_product = torch.einsum("ikp,jkp->ijk", self.llrs, enrollment_activations)
        final_product = torch.einsum("nkp,pnk->nk", test_activations, enroll_product)
        result = torch.sum(final_product, dim=1)
        return result

    def __str__(self):
        repr_str = (
            f"{self.__class__.__name__}: drop_in={self.drop_in}, typicality_threshold="
            f"{self.typicality_threshold}, device={self.device}.\n"
            f"typicality={self.typicality.shape}\ndrop_out={self.dropout.shape}"
            f"\nllrs={self.llrs.shape}"
        )
        return repr_str


class SpeechLLRScorer(LLRScorer):

    def LR_00(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        No_din = 1 - self.drop_in
        return (1 + drop_out**2) / (
            typicality * (2 * drop_out * No_din + No_din**2 + drop_out**2)
        )

    def LR_01(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        No_din = 1 - self.drop_in
        No_dout = 1 - drop_out
        return (No_din * self.drop_in * typicality + drop_out * No_dout) / (
            typicality
            * (
                1
                + self.drop_in * typicality * drop_out
                + No_din * self.drop_in * typicality
                + drop_out * No_dout
            )
        )

    def LR_10(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        return self.LR_01(typicality, drop_out)

    def LR_11(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        No_dout = 1 - drop_out
        return (1 + (self.drop_in * typicality) ** 2) / (
            typicality
            * (
                2 * self.drop_in * typicality * No_dout
                + (self.drop_in * typicality) ** 2
                + No_dout**2
            )
        )


class DNALLRScorer(LLRScorer):

    def LR_00(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        return 1 / (typicality * ((1 - self.drop_in) + drop_out))

    def LR_01(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        return self.drop_in / (self.drop_in * typicality + (1 - self.drop_in))

    def LR_10(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        return drop_out / typicality

    def LR_11(self, typicality: torch.Tensor, drop_out: torch.Tensor) -> torch.Tensor:
        return 1 / (typicality * (typicality * self.drop_in + (1 - drop_out)))


class MaxLLRScorer(Scorer):
    """
    This scorer also models the distribution of attributes over a reference population
    with the three parameters typicality, drop-out and drop-in, but instead of assuming
    that each speaker either possesses or does not possess each attribute, it assigns
    each speaker a probability of possessing or not possessing each attribute.

    This model is a more generic one than the DNA or Speech LLR scorers and it is based
    on standard statistical methods.

    To avoid confusion, the parameters (typicality, drop-out and drop-in) are renamed
    (f, p and q).
    """

    def __init__(
        self,
        n_attributes: int = 512,
        f: float = 0.5,
        p: float = 0.9,
        q: float = 0.1,
        checkpoint_path: str | Path | None = None,
        save_dir: str | Path | None = None,
        exist_ok: bool = False,
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(n_attributes, checkpoint_path, save_dir, exist_ok, device)
        if p < q:
            raise ValueError(
                "Invalid initial values for MaxLLRScorer. p must be greater than q."
            )
        self.f0 = f
        self.p0 = p
        self.q0 = q
        self.register_buffer("f", torch.zeros(n_attributes))
        self.register_buffer("p", torch.zeros(n_attributes))
        self.register_buffer("q", torch.zeros(n_attributes))
        self.load_model()

    def _fit_prior(self, referential: Tensor, n_iterations: int, *args, **kwargs) -> None:
        """
        Estimate scorer parameters from the activation statistics of a referential.

        Args:
            referential (torch.Tensor): a tensor of activation statistics of dimension
                N x n x K where N is the number of elements of the referential,
                n is the number of attributes, and K is the number of possible values
                for each attribute (i.e. 2 in case of binary attributes).
            n_iterations (int): number of iterations for parameter estimations.
        """
        check_tensor_shape(referential, self.n_attributes, 2, "Referential")

        LOGGER.info(
            "Estimating parameters with Expectation-Maximization "
            f"({n_iterations} iterations)."
        )

        f = torch.full(
            (self.n_attributes,), self.f0, dtype=torch.float64, device=self.device
        )
        p = torch.full(
            (self.n_attributes,), self.p0, dtype=torch.float64, device=self.device
        )
        q = torch.full(
            (self.n_attributes,), self.q0, dtype=torch.float64, device=self.device
        )

        activations = referential[:, :, 0]
        non_activations = referential[:, :, 1]
        for _ in range(n_iterations):
            f = (
                f
                * p**activations
                * (1 - p) ** non_activations
                / (
                    f * p**activations * (1 - p) ** non_activations
                    + (1 - f) * q**activations * (1 - q) ** non_activations
                )
            )

            p = (activations * f).sum(dim=0) / ((activations + non_activations) * f).sum(
                dim=0
            )
            q = (activations * (1 - f)).sum(dim=0) / (
                (activations + non_activations) * (1 - f)
            ).sum(dim=0)
            f = f.mean(dim=0)

        self.f = f
        self.p = p
        self.q = q

    def _likelihood(
        self, activations: torch.Tensor, non_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute vectorized LLR values for each attribute given a vector of activation
        values and a vector of non activation values.

        Args:
            activations (torch.Tensor): activations tensor of shape N x n
            non_activations (torch.Tensor): non activations tensor of shape N x n

        Returns:
            torch.Tensor: tensor of LLR values of shape N x n
        """
        return self.f * (self.p**activations) * ((1 - self.p) ** non_activations) + (
            1 - self.f
        ) * (self.q**activations) * ((1 - self.q) ** non_activations)

    def _score(self, enrollment_activations: Tensor, test_activations: Tensor) -> Tensor:
        check_tensor_shape(enrollment_activations, self.n_attributes, 2, "Enrollment")
        check_tensor_shape(test_activations, self.n_attributes, 2, "Test")

        ae = enrollment_activations[:, :, 0]  # enrollment activations
        ne = enrollment_activations[:, :, 1]  # enrollment non activations
        at = test_activations[:, :, 0]  # test activations
        nt = test_activations[:, :, 1]  # test non activations
        ratios = (
            torch.log(self._likelihood(ae + at, ne + nt))
            - torch.log(self._likelihood(ae, ne))
            - torch.log(self._likelihood(at, nt))
        )
        return torch.sum(ratios, dim=1)

    def __str__(self):
        repr_str = (
            f"{self.__class__.__name__}: f0={self.f0}, p0={self.p0}, "
            f"q0={self.q0}, device={self.device}.\n"
            f"f={self.f.shape}\np={self.p.shape}\nq={self.q.shape}."
        )
        return repr_str
