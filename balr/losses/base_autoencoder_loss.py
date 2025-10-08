from abc import ABC, abstractmethod

import torch


class BaseAutoEncoderLoss(torch.nn.Module, ABC):

    def __init__(self, name: str, weight: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_ = name
        self.weight_ = weight

    @property
    def weight(self) -> torch.Tensor:
        return (
            self.weight_
            if self.weight_ is not None
            else torch.tensor([1.0], dtype=torch.float)
        )

    @property
    def name(self) -> str:
        return self.name_

    def setup(self, **kwargs) -> None:
        """
        Setup function to initialize loss function with parameters that are known only
        when trainer is instantiated. The trainer will call setup on each loss function
        with named parameters that can be used to setup the loss.
        """
        pass

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        labels: torch.Tensor,
        output: torch.Tensor,
        recon: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            input (torch.Tensor): input features fed to the model (embeddings).
                torch.Tensor of size (batch_size, embedding_dim).
            labels (torch.Tensor): labels associated to the input.
                torch.Tensor of size (batch_size).
            output (torch.Tensor): model's encoded reprensentation of the input.
                torch.Tensor of size (batch_size, encoder_dim).
            recon (torch.Tensor): the reconstructed input from the model.
                torch.Tensor of size (batch_size, embedding_dim).
            Z (torch.Tensor): the model's latent space representation of the input.
                torch.Tensor of size (batch_size, encoder_dim).

        Returns:
            torch.Tensor: the loss.
        """
        pass
