from abc import ABC, abstractmethod

import numpy.typing as npt

from balr.data.dataset import AudioDataset


class EmbeddingsModel(ABC):
    """
    Abstract base class for Embedding Extraction models.
    """

    @abstractmethod
    def extract_embeddings(
        self, dataset: AudioDataset, stream_save: bool, *args, **kwargs
    ) -> list[tuple[str, npt.NDArray]]:
        pass
