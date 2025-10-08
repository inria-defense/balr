from pathlib import Path

import torch

from balr.scoring.scoring import Scorer


class CosineSimilarityScorer(Scorer):

    def __init__(
        self,
        dim: int = 1,
        eps: float = 1e-8,
        n_attributes: int = 512,
        checkpoint_path: str | Path | None = None,
        save_dir: str | Path | None = None,
        exist_ok: bool = False,
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(n_attributes, checkpoint_path, save_dir, exist_ok, device)

        self.model = torch.nn.CosineSimilarity(dim=dim, eps=eps)
        self.loaded = True

    def _score(
        self, enrollment_activations: torch.Tensor, test_activations: torch.Tensor
    ) -> torch.Tensor:
        return torch.prod(self.model(enrollment_activations, test_activations), dim=1)

    def _fit_prior(
        self, referential: torch.Tensor, n_iterations: int, *args, **kwargs
    ) -> None:
        raise NotImplementedError(
            "CosineSimilarityScorer does not need to be fit on reference population."
        )
