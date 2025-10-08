import torch

from balr.losses.base_autoencoder_loss import BaseAutoEncoderLoss


class MSELoss(BaseAutoEncoderLoss):

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        super().__init__("MSE", weight, *args, **kwargs)
        self.criterion = torch.nn.MSELoss()

    def forward(
        self,
        input: torch.Tensor,
        labels: torch.Tensor,
        output: torch.Tensor,
        recon: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.criterion(recon, input)
        return loss
