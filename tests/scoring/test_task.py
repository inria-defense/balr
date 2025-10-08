from pathlib import Path

import numpy.typing as npt
import pytest
import torch

from balr.data.dataset import AudioDataset
from balr.scoring.task import get_activation_tensors
from tests.conftest import BA_DIM


def test_get_activation_tensors_raises_error(voxceleb2_metadata: Path):
    dataset = AudioDataset.from_path(voxceleb2_metadata)

    with pytest.raises(ValueError, match="Binary Attributes Vector for"):
        get_activation_tensors(dataset, [["id03184/00073"]])


def test_get_activation_tensors_returns_correct_tensors(
    voxceleb2_metadata: Path, voxceleb2_ba_files: dict[str, npt.NDArray]
):
    dataset = AudioDataset.from_path(voxceleb2_metadata)

    ids = [
        ["id03184/00073", "id03184/00022", "id03184/00053"],
        ["id00906/00029"],
        ["id03701/00128", "id03701/00001"],
    ]
    activations = get_activation_tensors(dataset, ids)

    assert activations.shape == (3, BA_DIM, 2)

    for idx in range(len(ids)):
        expected_activations = torch.stack(
            [torch.from_numpy(voxceleb2_ba_files[f"{id}_ba.txt"]) for id in ids[idx]]
        ).sum(dim=0)
        assert torch.equal(activations[idx, :, :], expected_activations)
