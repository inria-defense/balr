from collections.abc import Generator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

ROOT_DIR = Path(__file__).parent.parent
EMBEDDING_DIM = 256
BA_DIM = 512


@pytest.fixture
def voxceleb1_dir() -> Path:
    return ROOT_DIR / "resources/data/voxceleb1"


@pytest.fixture
def voxceleb2_dir() -> Path:
    return ROOT_DIR / "resources/data/voxceleb2"


@pytest.fixture
def voxceleb2_metadata() -> Path:
    return ROOT_DIR / "resources/data/voxceleb2/metadata.csv"


def create_embedding_files(dir: Path) -> dict[str, npt.NDArray]:
    """
    Create embedding files for all audio files in directory.

    Args:
        dir (Path): the directory.

    Returns:
        dict[str, npt.NDArray]: a dict of embedding files.
    """
    embeddings = {}
    for file in dir.glob("**/*.wav"):
        emb = np.random.rand(256)
        emb_file = file.with_name(file.stem + "_emb").with_suffix(".txt")
        np.savetxt(emb_file, emb)
        embeddings[emb_file.name] = emb
    return embeddings


@pytest.fixture
def voxceleb2_embedding_files(voxceleb2_dir: Path) -> Generator[dict[str, npt.NDArray]]:
    """
    Fixture to generate random embeddings for audio files in dataset.

    Args:
        voxceleb2_dir (Path): path to the dataset dir

    Yields:
        Generator[dict[str, npt.NDArray]]: dict of embeddings
    """
    # remove all ".txt" files in dataset directory
    for file in voxceleb2_dir.glob("**/*.txt"):
        file.unlink()

    # create embedding files for all audio files in dataset directory
    embeddings = create_embedding_files(voxceleb2_dir)
    yield embeddings

    # remove all ".txt"  and ".pkl" files in dataset directory
    for file in voxceleb2_dir.glob("**/*.txt"):
        file.unlink()
    for file in voxceleb2_dir.glob("**/*.pkl"):
        file.unlink()


@pytest.fixture
def voxceleb1_embedding_files(voxceleb1_dir: Path) -> Generator[dict[str, npt.NDArray]]:
    """
    Generate embedding files for the voxceleb1 dataset. Then remove them after the
    test succeeded.

    Args:
        voxceleb1_dir (Path): path to the voxceleb1 directory.

    Yields:
        Generator[dict[str, npt.NDArray]]: dict of filename -> embedding vector.
    """
    # remove all ".txt" files in voxceleb_dir
    for file in voxceleb1_dir.glob("**/*.txt"):
        file.unlink()

    # create embedding files for all audio files in voxceleb_dir
    embeddings = create_embedding_files(voxceleb1_dir)

    yield embeddings

    # remove all ".txt" and ".pkl" files in voxceleb_dir
    for file in voxceleb1_dir.glob("**/*.txt"):
        file.unlink()
    for file in voxceleb1_dir.glob("**/*.pkl"):
        file.unlink()


@pytest.fixture
def voxceleb2_ba_files(voxceleb2_dir: Path) -> Generator[dict[str, npt.NDArray]]:
    """
    Generate BA vector files for the voxceleb2 dataset. Then remove them after the test
    succeeded.

    Args:
        voxceleb2_dir (Path): path to the voxceleb2 directory.

    Yields:
        Generator[dict[str, npt.NDArray]]: dict of filename -> BA vector.
    """
    # remove all ".txt" files in voxceleb2_dir
    for file in voxceleb2_dir.glob("**/*.txt"):
        file.unlink()

    # create ba files for all audio files in voxceleb_dir
    ba_files = {}
    for file in voxceleb2_dir.glob("**/*.wav"):
        ba = np.random.randint(2, size=BA_DIM)
        noba = np.absolute(ba - 1.0)
        ba = np.stack((ba, noba), axis=-1)
        ba_file = file.with_name(file.stem + "_ba").with_suffix(".txt")
        np.savetxt(ba_file, ba)
        ba_files[str(ba_file.relative_to(voxceleb2_dir))] = ba
    yield ba_files

    # remove all ".txt" and ".pkl" files in voxceleb_dir
    for file in voxceleb2_dir.glob("**/*.txt"):
        file.unlink()
    for file in voxceleb2_dir.glob("**/*.pkl"):
        file.unlink()
