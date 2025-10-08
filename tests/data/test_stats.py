from pathlib import Path

import numpy.typing as npt
import pytest

from balr.config.configs import DataConfig
from balr.data.dataset import AudioDataset
from balr.data.stats import get_speaker_statistics
from tests.conftest import BA_DIM


def test_get_statistics_returns_tensor(
    voxceleb2_ba_files: dict[str, npt.NDArray], voxceleb2_metadata: Path
):
    dataset = AudioDataset.from_path(voxceleb2_metadata)

    stats = get_speaker_statistics(dataset, DataConfig())
    assert stats.shape == (9, BA_DIM, 2)  # nb of speakers, nb of attributes, 2


def test_get_statistics_raises_error_if_no_bas(voxceleb2_metadata: Path):
    dataset = AudioDataset.from_path(voxceleb2_metadata)

    with pytest.raises(
        ValueError,
        match="An item in the dataset does not have a binary attributes vector.",
    ):
        get_speaker_statistics(dataset, DataConfig())
