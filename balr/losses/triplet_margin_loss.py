import torch
from pytorch_metric_learning import losses, miners

from balr.losses.base_autoencoder_loss import BaseAutoEncoderLoss


class TripletMarginLoss(BaseAutoEncoderLoss):

    def __init__(
        self,
        margin: float = 0.3,
        type_of_triplets: str = "all",
        weight: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        super().__init__("Triplet", weight, *args, **kwargs)
        self.margin = margin
        self.type_of_triplets = type_of_triplets
        self.miner = miners.TripletMarginMiner(
            margin=margin, type_of_triplets=type_of_triplets
        )
        self.criterion = losses.TripletMarginLoss()

    def forward(
        self,
        input: torch.Tensor,
        labels: torch.Tensor,
        output: torch.Tensor,
        recon: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        hard_pairs = self.miner(output, labels)
        loss = self.criterion(output, labels, hard_pairs)
        return loss
