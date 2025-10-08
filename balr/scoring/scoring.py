import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from balr.utils import select_device, setup_save_dir

LOGGER = logging.getLogger(__name__)


class Scorer(ABC, torch.nn.Module):
    """
    Abstract base class for BALR Scorers.
    """

    def __init__(
        self,
        n_attributes: int = 512,
        checkpoint_path: str | Path | None = None,
        save_dir: str | Path | None = None,
        exist_ok: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super(Scorer, self).__init__()
        self.n_attributes = n_attributes
        self.exist_ok = exist_ok

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        self.save_dir = save_dir

        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path = checkpoint_path

        self.device = select_device(device)
        self.loaded = False

    def fit(self, referential: torch.Tensor, n_iterations: int, *args, **kwargs) -> None:
        """
        Estimate scorer parameters from the activation statistics of a referential.

        Args:
            referential (Tensor): a tensor of activation statistics of dimension
                N x n x K where N is the number of elements of the referential,
                n is the number of attributes, and K is the number of possible values
                for each attribute (i.e. 2 in case of binary attributes).
            n_iterations (int): number of iterations for parameter estimations.
        """
        LOGGER.info(
            "<================ Scorer Training ================>\n"
            f"{self}\nReferential: shape={referential.shape}\n"
            "<=================================================>"
        )
        start_time = time.time()

        referential = referential.to(self.device)
        self._fit_prior(referential=referential, n_iterations=n_iterations)
        self.loaded = True

        for name, buffer in self.named_buffers():
            if torch.any(torch.isnan(buffer)):
                LOGGER.error(
                    f"Estimated {name} parameters contain NaN values. "
                    "Please refit the scorer on a larger dataset."
                )

        seconds = time.time() - start_time
        LOGGER.info(f"Scorer training completed in {seconds / 3600:.3f} hours.")

    def forward(
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
        if not self.loaded:
            LOGGER.error(
                "Model weights have not been initialized. Make sure to fit the scorer "
                "on a reference dataset to estimate its parameters, and to run the "
                "scorer with the weights saved."
            )

        enrollment_activations = enrollment_activations.to(self.device)
        test_activations = test_activations.to(self.device)

        LOGGER.info(
            "<==================== Scoring ====================>\n"
            f"{self}\nEnrollments: shape={enrollment_activations.shape}\n"
            f"Tests: shape={test_activations.shape}\n"
            "<=================================================>"
        )

        return self._score(
            enrollment_activations=enrollment_activations,
            test_activations=test_activations,
        )

    @abstractmethod
    def _fit_prior(
        self, referential: torch.Tensor, n_iterations: int, *args, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def _score(
        self, enrollment_activations: torch.Tensor, test_activations: torch.Tensor
    ) -> torch.Tensor:
        pass

    def load_model(self):
        """
        Load model weights from disk.
        """
        if self.checkpoint_path is not None:
            LOGGER.info(f"Loading scorer weights from {self.checkpoint_path}.")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint, strict=False)
            self.to(self.device)
            self.loaded = True

    def save_model(self):
        """
        Save model weights.
        """
        self.save_dir = setup_save_dir(
            self.save_dir, exist_ok=self.exist_ok, default_path="runs/scorer/train"
        )

        if self.save_dir is not None:
            LOGGER.info("Saving scorer weights...")
            fname = self.save_dir / "scorer.pt"
            torch.save(self.state_dict(), fname)
            LOGGER.info(f"Scorer weights saved to [bold blue]{fname}[/].")


def check_tensor_shape(tensor: torch.Tensor, n: int, K: int, name: str) -> None:
    """
    Check that tensor is of shape N x n x K.

    Args:
        tensor (torch.Tensor): the tensor to check.
        n (int): second dim.
        K (int): third dim.
        name (str): name of the tensor.

    Raises:
        ValueError: if tensor is not of shape N x n x K.
    """
    if len(tensor.shape) != 3:
        raise ValueError(
            f"{name} tensor should be of shape N x n x K but instead has "
            f"shape {tensor.shape}."
        )

    if tensor.shape[1] != n:
        raise ValueError(
            f"{name} tensor should be of shape N x n x K with "
            f"n={n} but instead has shape {tensor.shape}."
        )

    if tensor.shape[-1] != K:
        raise ValueError(
            f"{name} tensor should be of shape N x n x K with "
            f"K={K} but instead has shape {tensor.shape}."
        )
