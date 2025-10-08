import torch
from pytorch_metric_learning import losses, miners

from balr.losses.base_autoencoder_loss import BaseAutoEncoderLoss


class ArcFaceLoss(BaseAutoEncoderLoss):

    def __init__(
        self,
        margin: float = 0.3,
        type_of_triplets: str = "all",
        weight: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        super().__init__("ArcFace", weight, *args, **kwargs)
        self.margin = margin
        self.type_of_triplets = type_of_triplets
        self.miner = miners.TripletMarginMiner(
            margin=margin, type_of_triplets=type_of_triplets
        )
        self.criterion = None

    def setup(  # type: ignore[override]
        self,
        nb_train_classes: int,
        internal_dim: int,
        **kwargs,
    ):
        self.num_classes = nb_train_classes
        self.embedding_size = internal_dim
        self.criterion = losses.ArcFaceLoss(
            num_classes=nb_train_classes, embedding_size=internal_dim
        )

    def forward(
        self,
        input: torch.Tensor,
        labels: torch.Tensor,
        output: torch.Tensor,
        recon: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        if self.criterion is None:
            raise RuntimeError(
                "ArcFaceLoss function has not been properly setup. "
                "Make sure to call setup on instance."
            )

        hard_pairs = self.miner(output, labels)
        loss = self.criterion(output, labels, hard_pairs)
        return loss
