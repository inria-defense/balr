from abc import ABC, abstractmethod

import numpy.typing as npt

from balr.data.dataset import AudioDataset


class BinaryAttributeEncoder(ABC):
    """
    Abstract base class for Binary Attribute Encoders.
    """

    @abstractmethod
    def get_binary_attributes(
        self, dataset: AudioDataset, stream_save: bool, *args, **kwargs
    ) -> list[tuple[str, npt.NDArray]]:
        pass
