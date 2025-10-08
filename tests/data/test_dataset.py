import pickle
import shutil
from collections.abc import Generator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import torch
from numpy.testing import assert_equal

from balr.data.dataset import AudioDataset, VectorExtension, VectorType
from tests.conftest import ROOT_DIR


@pytest.fixture
def output_dir() -> Generator[Path]:
    output_dir = ROOT_DIR / "resources/data/voxcelebtest"

    yield output_dir

    shutil.rmtree(output_dir)


@pytest.fixture
def voxceleb1_test_file() -> tuple[str, int]:
    """
    Get test filename and tensor length

    Returns:
        tuple[str, int]: filename and tensor length
    """
    return "id10270+5r0dWxy17C8+00007.wav", 236161


def test_dataset_create_from_dir(
    voxceleb1_dir: Path, voxceleb1_test_file: tuple[str, int]
):
    dataset = AudioDataset.from_dir(voxceleb1_dir)
    assert len(dataset) == 3

    expected_file_name, expected_audio_frames = voxceleb1_test_file
    item = dataset[1]
    assert item.audio is not None
    assert item.audio.shape == (1, expected_audio_frames)
    assert item.sample_rate == dataset.sample_rate
    assert item.audio_path == voxceleb1_dir / expected_file_name

    with pytest.raises(RuntimeError):
        AudioDataset.from_dir(ROOT_DIR / "not_a_dir")


def test_dataset_create_from_file(
    voxceleb1_dir: Path, voxceleb1_test_file: tuple[str, int]
):
    expected_file_name, expected_audio_frames = voxceleb1_test_file
    dataset = AudioDataset.from_file(voxceleb1_dir / expected_file_name)

    assert len(dataset) == 1
    item = dataset[0]
    assert item.audio is not None
    assert item.audio.shape == (1, expected_audio_frames)
    assert item.sample_rate == dataset.sample_rate
    assert item.audio_path == voxceleb1_dir / expected_file_name


def test_dataset_loads_embeddings(
    voxceleb1_dir: Path,
    voxceleb1_test_file: tuple[str, int],
    voxceleb1_embedding_files: dict[str, npt.NDArray],
):
    dataset = AudioDataset.from_dir(voxceleb1_dir)
    assert len(dataset) == 3

    expected_file_name, _ = voxceleb1_test_file
    embedding_file = expected_file_name[:-4] + "_emb.txt"
    item = dataset[1]
    assert item.embedding is not None
    assert_equal(item.embedding.numpy(), voxceleb1_embedding_files[embedding_file])


def test_dataset_save_vectors(
    voxceleb1_dir: Path, voxceleb1_embedding_files: dict[str, npt.NDArray]
):
    # Create dataset from directory without BA vectors
    dataset = AudioDataset.from_dir(voxceleb1_dir)
    assert len(dataset) == 3
    item = dataset[2]
    assert item.binary_attributes is None

    # Create BA vector for item in dataset
    ba = np.random.rand(512, 2)
    attributes = [(item.id, ba)]
    dataset.save_vectors(attributes, VectorType.binary_attributes, VectorExtension.txt)

    # Ensure file exists on disk and BA vector is loaded when item is accessed
    assert item.audio_path is not None
    ba_file = item.audio_path.with_name(item.audio_path.stem + "_ba").with_suffix(".txt")
    assert ba_file.exists()
    item = dataset[2]
    assert item.binary_attributes is not None
    assert_equal(item.binary_attributes.numpy(), ba)


def test_dataset_save_vectors_as_pickle(
    voxceleb1_dir: Path, voxceleb1_embedding_files: dict[str, npt.NDArray]
):
    # Create dataset from directory without BA vectors
    dataset = AudioDataset.from_dir(voxceleb1_dir)
    assert len(dataset) == 3
    item = dataset[2]
    assert item.binary_attributes is None

    # Create BA vector for item in dataset
    ba = np.random.rand(512, 2)
    attributes = [(item.id, ba)]
    dataset.save_vectors(attributes, VectorType.binary_attributes, VectorExtension.pkl)

    # Ensure file exists on disk and BA vector is loaded when item is accessed
    assert item.audio_path is not None
    ba_file = item.audio_path.with_name(item.audio_path.stem + "_ba").with_suffix(".pkl")
    assert ba_file.exists()
    item = dataset[2]
    assert item.binary_attributes is not None
    assert_equal(item.binary_attributes.numpy(), ba)


def test_dataset_save_vector_with_output_dir(voxceleb2_metadata: Path, output_dir: Path):
    dataset = AudioDataset.from_path(voxceleb2_metadata)
    dataset.set_output_dir(output_dir)

    # Create BA vector for item in dataset
    ba = np.random.rand(512, 2)

    # Save BA vectors to output dir
    items = (dataset[2], dataset[1])
    dataset.save_vectors(
        [(item.id, ba) for item in items],
        VectorType.binary_attributes,
        VectorExtension.txt,
    )

    for item in items:
        assert item.audio_path is not None
        ba_file = output_dir / item.audio_path.relative_to(voxceleb2_metadata.parent)
        ba_file = ba_file.with_name(ba_file.stem + "_ba").with_suffix(".txt")
        assert ba_file.exists()


def test_dataset_create_from_csv(voxceleb2_metadata: Path):
    dataset = AudioDataset.from_csv(voxceleb2_metadata)
    dataset.set_normalize(True)
    assert len(dataset) == 27

    expected_filename = voxceleb2_metadata.parent / "id02477/00079.wav"
    expected_audio_frames = 114688
    expected_speaker = "id02477"
    item = dataset[7]
    assert item.audio is not None
    assert item.audio.shape == (1, expected_audio_frames)
    assert item.audio_path == expected_filename
    assert item.speaker == expected_speaker


def test_dataset_with_duplicate_ids(voxceleb1_dir: Path):
    dataset = AudioDataset.from_dir(voxceleb1_dir)
    assert len(dataset) == 3

    dataset = AudioDataset(
        dataset.data, dataset.data_ids + dataset.data_ids[0:2], sort=False
    )
    assert len(dataset) == 5

    assert dataset[3].id == dataset[0].id
    assert torch.equal(dataset[3].audio, dataset[0].audio)  # type: ignore[arg-type]
    assert dataset[4].id == dataset[1].id
    assert torch.equal(dataset[4].audio, dataset[1].audio)  # type: ignore[arg-type]


def test_dataset_filter(voxceleb1_dir: Path):
    dataset = AudioDataset.from_dir(voxceleb1_dir)
    assert len(dataset) == 3

    dataset = AudioDataset(
        dataset.data, dataset.data_ids + dataset.data_ids[0:2], sort=False
    )
    assert len(dataset) == 5

    # keep only items with id id10270+5r0dWxy17C8+00004
    dataset = dataset.filter(lambda item: item[0].endswith("00004.wav"))
    assert len(dataset) == 2
    assert dataset[0].id == "id10270+5r0dWxy17C8+00004.wav"
    assert dataset[1].id == "id10270+5r0dWxy17C8+00004.wav"


def test_dataset_str_repr(
    voxceleb2_metadata: Path,
    voxceleb1_dir: Path,
    voxceleb1_embedding_files: dict[str, npt.NDArray],
):
    dataset = AudioDataset.from_dir(voxceleb1_dir)
    assert str(dataset) == (
        f"AudioDataset: size={len(dataset)}, sample_rate={dataset.sample_rate}, "
        f"embeddings={len(voxceleb1_embedding_files)}, binary_attributes=0, classes=0."
    )

    dataset = AudioDataset.from_path(voxceleb2_metadata)
    assert str(dataset) == (
        f"AudioDataset: size={len(dataset)}, sample_rate={dataset.sample_rate}, "
        f"embeddings=0, binary_attributes=0, classes=9."
    )


def test_dataset_loads_without_audio_with_only_embeddings(
    voxceleb1_dir: Path, voxceleb1_embedding_files: dict[str, npt.NDArray]
):
    # Create metadata.csv file with ids and embedding values
    metadata_csv = voxceleb1_dir / "embeddings_only.csv"
    try:
        records = [
            {
                "id": name[:-8],
                "embedding": voxceleb1_dir / name,
                "speaker": name[: name.find("+")],
            }
            for name in voxceleb1_embedding_files.keys()
        ]
        df = pd.DataFrame.from_records(records)
        df.to_csv(metadata_csv, index=False)

        # Create AudioDataset from csv file
        dataset = AudioDataset.from_path(metadata_csv)
        assert len(dataset) == 3

        item = dataset[1]
        assert item.audio is None
        assert item.embedding is not None
        assert item.embedding.shape == (256,)
        assert item.speaker == records[1]["speaker"]
    finally:
        if metadata_csv.exists():
            metadata_csv.unlink()


def test_dataset_loads_pickle_vectors(
    voxceleb1_dir: Path, voxceleb1_embedding_files: dict[str, npt.NDArray]
):
    # Delete .txt embedding file and resave it as pickle
    name, embedding = voxceleb1_embedding_files.popitem()
    fp = voxceleb1_dir / name
    fp.unlink()
    with open(fp.with_suffix(".pkl"), "wb") as f:
        pickle.dump(embedding, f)

    # Create AudioDataset from dir
    dataset = AudioDataset.from_path(voxceleb1_dir)
    assert len(dataset) == 3

    for idx in range(len(dataset)):
        item = dataset[idx]
        assert item.embedding is not None
        assert item.embedding.shape == (256,)
