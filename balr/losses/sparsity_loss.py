import numpy as np
import numpy.typing as npt
import torch

from balr.losses.base_autoencoder_loss import BaseAutoEncoderLoss


class SparsityLoss(BaseAutoEncoderLoss):

    def __init__(self, weight: torch.Tensor | None = None, *args, **kwargs):
        super().__init__("Sparsity", weight, *args, **kwargs)
        self.target: npt.NDArray | None = None

    def setup(  # type: ignore[override]
        self,
        M_samples_per_class: int,
        internal_dim: int,
        **kwargs,
    ):
        self.encoder_dim = internal_dim
        self.M_samples_per_class = M_samples_per_class
        self.target = np.random.randint(0, M_samples_per_class, size=(self.encoder_dim,))

    def forward(
        self,
        input: torch.Tensor,
        labels: torch.Tensor,
        output: torch.Tensor,
        recon: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        if self.target is None:
            raise RuntimeError(
                "SparsityLoss function has not been properly setup. "
                "Make sure to call setup on instance."
            )

        # Group the activations of all x-vectors per speaker, and sum them in batches
        # of M_samples_per_class
        speakers = labels.unique()
        activation_sums = []
        for speaker in speakers:
            speaker_indices = torch.where(labels == speaker)
            speaker_activations = Z[speaker_indices].detach().cpu().numpy()

            # Check that the number of samples for speaker is divisible by
            # M_samples_per_class.
            if len(speaker_activations) % self.M_samples_per_class:
                raise ValueError(
                    f"Error computing sparsity loss. Speaker with label {speaker} has "
                    f"{len(speaker_activations)} utterances in batch which is not "
                    "divisible by `M_samples_per_class` param ("
                    f"{self.M_samples_per_class})."
                )

            # split speaker_activations into batches of M_samples_per_class
            for speaker_batch in np.split(
                speaker_activations, len(speaker_activations) // self.M_samples_per_class
            ):
                activation_sums.append(speaker_batch.sum(0))

        activation_sums = np.stack(activation_sums)
        loss = (
            torch.from_numpy(activation_sums - self.target)
            .requires_grad_(True)
            .clamp(0, 1)
            .pow(2)
            .sum()
        )

        loss = loss / len(activation_sums)
        return loss
