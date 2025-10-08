"""
Architecture of the binary autoencoder of the following paper.
 I. Ben-Amor, J.-F. Bonastre, et S. Mdhaffar, « Extraction of interpretable
 and shared speaker-specific speech attributes through binary auto-encoder »,
 in Interspeech 2024, ISCA, sept. 2024, p. 3230-3234. doi: 10.21437/Interspeech.2024-1011.

Code fourni par Imen Ben-Amor.
"""

import logging
import time
from pathlib import Path

import numpy.typing as npt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from balr.binary_attributes.encoder import BinaryAttributeEncoder
from balr.config.configs import DataConfig
from balr.data.dataset import AnnotatedAudio, AudioDataset, VectorType, build_dataloader
from balr.utils import select_device

LOGGER = logging.getLogger(__name__)


def collate_fn(batch: list[AnnotatedAudio]) -> tuple[torch.Tensor, list[str]]:
    embeddings: list[torch.Tensor] = []
    ids = []
    for sample in batch:
        if sample.embedding is None:
            LOGGER.warning(
                f"Sample {sample.id} does not have an embedding. Binarization skipped."
            )
            continue

        embeddings.append(sample.embedding)
        ids.append(sample.id)

    return torch.stack(embeddings), ids


class STEFunction(autograd.Function):
    """
    Straight through estimator function.

    Forward compares the input to zero. Backward uses the hardtanh function.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            ctx (): default context argument.
            input (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: binary activation tensor (0 or 1) as float values.
        """
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Backward computation.

        Args:
            ctx (): default context argument
            grad_output (torch.Tensor): output gradient.

        Returns:
            torch.Tensor: input tensor.
        """
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    """
    Straight through estimator object.

    Uses STEFunction.
    """

    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: binary activation tensor (0 or 1) as float values.
        """
        return STEFunction.apply(x)


class AutoEncoder(nn.Module):
    """
    Binary encoder

    Attributes:
        input_dim (int, default=256): dimension of input features
        internal_dim (int, default=512): dimension of the latent binarized space
        encoder (torch.nn.Module): encoder
        binarization (StraightThroughEstimator): binarization layer
    """

    def __init__(self, input_dim: int = 256, internal_dim: int = 512):
        """
        Initialize Binary Encoder

        Args:
            input_dim (int, optional): dimension of input features. Defaults to 256.
            internal_dim (int, optional): dimension of the latent binarized space.
                Defaults to 512.
        """
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.internal_dim = internal_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, internal_dim),
            nn.ReLU(),
            nn.BatchNorm1d(internal_dim),
            nn.Linear(internal_dim, internal_dim),
            nn.Tanh(),
        )
        self.binarization = StraightThroughEstimator()
        self.decoder = nn.Sequential(
            nn.Linear(internal_dim, internal_dim),
            nn.Tanh(),
            nn.Linear(internal_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method.

        Args:
            x (torch.Tensor): input_tensor of dimension (N, input_dim)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of tensors, including
                recon, binary activations corresponding to counts of ones and zeros
                (N, latent dim, 2), and the encoded sparse representation of input vector.
        """
        Z = self.encoder(x)
        binary = self.binarization(Z)
        recon = self.decoder(binary)
        return recon, binary, Z


class BinaryAttributeAutoEncoder(BinaryAttributeEncoder):
    def __init__(
        self,
        checkpoint_path: str | Path,
        input_dim: int = 256,
        internal_dim: int = 512,
        device: str | torch.device = "cpu",
    ):
        """
        Initialize the BinaryAttribute AutoEncoder class, to perform inference using
        an AutoEncoder model.

        Args:
            checkpoint_path (str | Path): model checkpoint.
            input_dim (int, optional): dimension of input features. Defaults to 256.
            internal_dim (int, optional): dimension of the latent binarized space.
                Defaults to 512.
            device (str | torch.device, optional): device to run predictions on.
                Defaults to "cpu".
        """
        self.device = select_device(device)
        self.checkpoint_path = checkpoint_path

        LOGGER.info(f"Loading AutoEncoder model checkpoint: {checkpoint_path}.")
        self.model = AutoEncoder(input_dim, internal_dim)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device).eval()

    def get_binary_attributes(
        self, dataset: AudioDataset, stream_save: bool, data_config: DataConfig
    ) -> list[tuple[str, npt.NDArray]]:
        dataloader = build_dataloader(
            dataset,
            data_config.batch_size,
            data_config.num_workers,
            data_config.shuffle,
            -1,
            self.device,
            collate_fn,
        )

        LOGGER.info(
            "<================ Binary Attribute Inference ================>\n"
            f"{self}\n{dataset}\n"
            f"DataLoader: batch_size={dataloader.batch_size}, num_batches="
            f"{len(dataloader)}, workers={dataloader.num_workers}\n"
            "<============================================================>"
        )
        start_time = time.time()

        binary_attributes: list[tuple[str, npt.NDArray]] = []
        for embeddings, ids in tqdm(dataloader):
            outputs = self._get_binary_activations(embeddings)
            ba = outputs.cpu().detach().numpy()

            batch_ba = list(zip(ids, ba))
            if stream_save:
                dataset.save_vectors(batch_ba, vector_type=VectorType.binary_attributes)
            else:
                binary_attributes.extend(batch_ba)

        seconds = time.time() - start_time
        LOGGER.info(
            f"Binary Attribute inference completed in {seconds / 3600:.3f} hours."
        )
        return binary_attributes

    def _get_binary_activations(self, embedding_vector: torch.Tensor) -> torch.Tensor:
        """
        Binarization of the embedding vector.

        Args:
            embedding_vector (torch.Tensor): embedding vector.

        Returns:
            torch.Tensor: binary activations corresponding to counts of ones and zeros
                (N, latent dim, 2)
        """
        if len(embedding_vector.shape) < 2:
            embedding_vector = embedding_vector.unsqueeze(0)
        _, binary, _ = self.model(embedding_vector.to(torch.float).to(self.device))
        result = torch.stack((binary, 1 - binary), dim=-1)
        return result

    def __str__(self):
        repr_str = (
            f"{self.__class__.__name__}: input_dim={self.model.input_dim}, internal_dim="
            f"{self.model.internal_dim}, checkpoint={self.checkpoint_path}, device="
            f"{self.device}."
        )
        return repr_str
