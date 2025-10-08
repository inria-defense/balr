from collections.abc import Generator
from pathlib import Path

import pytest
import torch

from balr.binary_attributes.autoencoder import AutoEncoder, BinaryAttributeAutoEncoder
from balr.binary_attributes.trainer import init_weights
from balr.config.configs import DataConfig
from balr.data.dataset import AudioDataset
from tests.conftest import BA_DIM, EMBEDDING_DIM, ROOT_DIR


@pytest.fixture
def model_checkpoint() -> Generator[Path]:
    checkpoint_path = ROOT_DIR / "resources/models/mock_random_weights.pt"
    model = AutoEncoder(EMBEDDING_DIM, BA_DIM)
    model.apply(init_weights)
    torch.save(model.state_dict(), checkpoint_path)

    yield checkpoint_path

    checkpoint_path.unlink()


@pytest.fixture
def voxceleb2_dataset() -> AudioDataset:
    metadata = ROOT_DIR / "resources/data/voxceleb2/metadata.csv"
    dataset = AudioDataset.from_csv(metadata)
    return dataset


def test_ba_autoencoder(voxceleb2_embedding_files, voxceleb2_dataset, model_checkpoint):
    encoder = BinaryAttributeAutoEncoder(model_checkpoint, EMBEDDING_DIM, BA_DIM)

    binary_attributes = encoder.get_binary_attributes(
        voxceleb2_dataset, False, DataConfig(batch_size=5, num_workers=0)
    )

    assert len(binary_attributes) == len(voxceleb2_dataset)
    for _, ba in binary_attributes:
        assert ba.shape == (BA_DIM, 2)
