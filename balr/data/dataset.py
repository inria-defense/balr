from __future__ import annotations

import logging
import os
import pickle
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, distributed


class VectorType(str, Enum):
    embedding = "embedding"
    binary_attributes = "binary_attributes"


class VectorExtension(str, Enum):
    txt = ".txt"
    pkl = ".pkl"


VECTOR_SUFFIXES = {
    VectorType.embedding: "_emb",
    VectorType.binary_attributes: "_ba",
}

LOGGER = logging.getLogger(__name__)


def load_audio(
    file: Path, target_sample_rate: int, normalize: bool = False
) -> torch.Tensor:
    """
    Load an audio file and resample to target sample rate if needed.
    Ensures loaded tensor is mono chanel.

    Args:
        file (Path): path to the audio file.
        target_sample_rate (int): target sample rate.
        normalize (bool, optional):
            When ``True``, this function converts the native sample type to ``float32``.
            Default: ``True``.

            If input file is integer WAV, giving ``False`` will change the resulting
            Tensor type to integer type.
            This argument has no effect for formats other than integer WAV type.

    Returns:
        torch.Tensor: the audio signal.
    """
    waveform, _sample_rate = torchaudio.load(file, normalize=normalize)

    # ensure sample_rate == dataset sample rate
    if _sample_rate != target_sample_rate:
        transform = T.Resample(_sample_rate, target_sample_rate)
        waveform = transform(waveform)

    # ensure signal is mono chanel
    if len(waveform.shape) > 1 and waveform.shape[0] > 1:
        # Do a mean of all channels and keep it in one channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform


def load_vector_file(vector_file: Path) -> torch.Tensor:
    """
    Load a vector from a file on disk. A vector can be saved either as a numpy array
    in txt format or pickle.

    Args:
        vector_file (Path): a .txt or .pkl file containing a numpy array

    Raises:
        ValueError: if the file has an unknown extension.

    Returns:
        torch.Tensor: the loaded vector.
    """
    if vector_file.suffix == ".txt":
        array = np.loadtxt(vector_file)
    elif vector_file.suffix == ".pkl":
        try:
            with open(vector_file, "rb") as f:
                array = pickle.load(f)
        except EOFError:
            raise EOFError(
                f"Unable to load vector from file {vector_file}. "
                "File appears to be empty."
            )
    else:
        raise ValueError(
            f"Unsupported file extension {vector_file.suffix} for {vector_file}."
        )
    return torch.from_numpy(array)


def vector_file_name(
    audio_file: Path,
    vector_type: VectorType,
    vector_extension: VectorExtension = VectorExtension.txt,
) -> Path:
    """
    Get the file name for the vector type of the given audio file.

    Example:
        > vector_file_name("/path/to/spk1utt1.wav", "embedding")
        /path/to/spk1utt1_emb.txt

    Args:
        audio_file (Path): path to the audio file.
        vector_type (VectorType): vector type.
        vector_extension (VectorExtension, optional): vector extension.
            Defaults to VectorExtension.txt.

    Returns:
        Path: path to the vector file.
    """
    suffix = VECTOR_SUFFIXES[vector_type]
    return audio_file.with_name(
        audio_file.stem.removesuffix(suffix) + suffix
    ).with_suffix(vector_extension.value)


def find_vector_files(audio_file: Path) -> dict[str, Path]:
    """
    Find all vector files corresponding to the given audio file that exist on disk.
    A vector file must have the same path as the audio file, with a suffix corresponding
    to its type (i.e. an embedding file for `spk1utt1.wav` must be named
    `spk1utt1_emb`) and a supported extension (`.txt` or `.pkl`).

    Args:
        audio_file (Path): path to the audio file.

    Returns:
        dict[str, Path]: a dict of vector type -> file path.
    """
    files = {}
    for vector in VectorType:
        for extension in VectorExtension:
            vector_file = vector_file_name(audio_file, vector, extension)
            if vector_file.exists():
                files[vector.value] = vector_file
                break
    return files


def get_item_reference_path(id: str, item: dict[str, Any]) -> Path:
    """
    Get the reference path for an AudioDataset item. The reference path is the path of
    the audio file, the embedding file, or the binary_attribute file, whichever exists
    first.

    Args:
        id (str): the AudioDataset item id.
        item (dict[str, Any]): the AudioDataset item

    Raises:
        RuntimeError: if no audio, embedding or binary_attribute file exists for this
            item.

    Returns:
        Path: the reference path.
    """
    for key in ["audio"] + [vt.value for vt in VectorType]:
        if key in item and item[key] is not None:
            return item[key]

    raise RuntimeError(
        f"Invalid empty item with id {id}: {item}. "
        "You must supply at least an audio or a vector file path for each item "
        "in an AudioDataset."
    )


@dataclass
class AnnotatedAudio:
    id: str
    sample_rate: int
    audio_path: Path | None = None
    audio: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    binary_attributes: torch.Tensor | None = None
    speaker: str | int | None = None


class AudioDataset(torch.utils.data.Dataset):
    """
    A dataset of AnnotatedAudio items.

    Each item in the dataset is identified by an id and provides information for
    an audio file and optional computed attributes, such as an embedding vector,
    or binary attribute vectors.

    The dataset takes as input a dict of dicts as the collection of data points
    to read. The top level keys are data point IDs. Each data point dict should
    have the same keys, corresponding to different files in that data point.

    For example the input data could look like this

    >>> data = {
    ...  "spk1utt1": {
    ...      "audio": "/path/to/spk1utt1.wav",
    ...      "embedding": "/path/to/spk1utt1_emb.txt",
    ...      "binary_attributes": "/path/to/spk1utt1_ba.txt",
    ...      },
    ...  "spk1utt2": {
    ...      "wav_file": "/path/to/spk1utt2.wav",
    ...      "embedding": "/path/to/spk1utt2_emb.txt",
    ...      "binary_attributes": "/path/to/spk1utt2_ba.txt",
    ...      }
    ... }

    The files are only loaded when the dataset is iterated.
    """

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        audio_formats: list[str] = ["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"],
        sample_rate: int = 16_000,
        root_dir: Path | None = None,
    ) -> AudioDataset:
        if not isinstance(path, Path):
            path = Path(path)

        if not path.exists():
            raise RuntimeError(f"Path {path} does not exist.")

        if path.is_dir():
            return cls.from_dir(
                path, audio_formats=audio_formats, sample_rate=sample_rate
            )
        elif path.is_file() and path.suffix.lower() == ".csv":
            return cls.from_csv(path, sample_rate=sample_rate, root_dir=root_dir)
        else:
            return cls.from_file(path, sample_rate=sample_rate)

    @classmethod
    def from_csv(
        cls,
        csv_file: str | Path,
        sample_rate: int = 16_000,
        root_dir: Path | None = None,
    ) -> AudioDataset:
        """
        Create a dataset from a csv file.

        The csv file must contain an audio column
        or a vector column (either embedding or binary_attributes) that point to files
        on disk. The paths can be absolute or relative. If relative, they can be relative
        to the csv file (default) or to the provided `root_dir` param.

        The csv file must also containg either an `id` column, or an `audio` column which
        will be used as id if the `id` column is not provided.

        Args:
            csv_file (str | Path): the csv file.
            sample_rate (int, optional): target sample rate. Defaults to 16_000.
            root_dir (Path | None, optional): root path to use as prefix for relative
                paths in the csv. If None, the csv's parent directory is used.
                Defaults to None.

        Raises:
            RuntimeError: if dir does not exist.

        Returns:
            AudioDataset: the dataset containing elements from input directory.
        """
        if not isinstance(csv_file, Path):
            csv_file = Path(csv_file)

        # 1. read csv file
        df = pd.read_csv(csv_file, header=0)

        # 2. set index column use audio file name as id
        if "id" in df.columns:
            # use id column if present
            df = df.set_index("id")
        elif "audio" in df.columns:
            # use "audio" column if present
            df["idx"] = df["audio"]
            df = df.set_index("idx")
        else:
            raise RuntimeError(
                "Either an id or an audio column must be present "
                "in the csv metadata file."
            )

        # 3. transform audio and vector paths relative to root_dir (only if paths are
        # not absolute already).
        if root_dir is None:
            root_dir = csv_file.parent

        def transform_relative_to_root_dir(file: str):
            fp = Path(file)
            if fp.is_absolute():
                return fp
            return root_dir / fp

        for column in ["audio"] + [vt.value for vt in VectorType]:
            if column in df.columns:
                df[column] = df[column].map(transform_relative_to_root_dir)

        data = df.to_dict(orient="index")

        # 4. look for vector files matching the audio file if the audio column is present
        if "audio" in df.columns:
            for item in data.values():
                item.update(find_vector_files(item["audio"]))

        return cls(data, sample_rate=sample_rate)  # type: ignore

    @classmethod
    def from_file(cls, file: str | Path, sample_rate: int = 16_000) -> AudioDataset:
        """
        Create a dataset from a single audio file. Uses filename as item id.

        Args:
            file (str | Path): audio file path.
            sample_rate (int, optional): target sample rate. Defaults to 16_000.

        Returns:
            AudioDataset: the dataset containing only one item.
        """
        if not isinstance(file, Path):
            file = Path(file)

        if not file.is_file():
            raise RuntimeError(
                f"{file} is not a file. Use AudioDataset.from_dir instead."
            )

        data = {file.stem: {"audio": file, **find_vector_files(file)}}
        return cls(data, sample_rate=sample_rate)

    @classmethod
    def from_dir(
        cls,
        dir: str | Path,
        audio_formats: list[str] = ["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"],
        sample_rate: int = 16_000,
    ) -> AudioDataset:
        """
        Create a dataset from a directory of audio files. Uses filenames as ids.

        Args:
            dir (str | Path): the directory.
            audio_formats (list[str], optional): audio formats to load.
                Defaults to ["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"].
            sample_rate (int, optional): target sample rate. Defaults to 16_000.

        Raises:
            RuntimeError: if dir does not exist.

        Returns:
            AudioDataset: the dataset containing elements from input directory.
        """
        if not isinstance(dir, Path):
            dir = Path(dir)

        if not dir.is_dir():
            raise RuntimeError(f"Dataset directory {dir} does not exist.")

        file_extensions = {
            extension.lower() if extension.startswith(".") else ("." + extension.lower())
            for extension in audio_formats
        }
        files = [
            file for file in dir.glob("**/*") if file.suffix.lower() in file_extensions
        ]
        data = {
            str(file.relative_to(dir)): {"audio": file, **find_vector_files(file)}
            for file in files
        }
        return cls(data, sample_rate=sample_rate)

    def __init__(
        self,
        data: dict[str, dict[str, Any]],
        data_ids: list[str] | None = None,
        sample_rate: int = 16_000,
        sort: bool = True,
        output_dir: Path | None = None,
    ):
        self.data = data
        if data_ids is None:
            data_ids = list(self.data.keys())
        if sort:
            data_ids.sort()
        self.data_ids = data_ids
        self.sample_rate = sample_rate
        self.normalize = True
        self.set_output_dir(output_dir)

    def set_normalize(self, norm: bool):
        """
        When ``True``, this function converts the native sample type to ``float32``.
        Default: ``True``.

        If input file is integer WAV, giving ``False`` will change the resulting
        Tensor type to integer type.
        This argument has no effect for formats other than integer WAV type.
        """
        self.normalize = norm

    def set_output_dir(self, output_dir: Path | None):
        self.output_dir = output_dir
        self.common_path: Path | None = None

        if self.output_dir is not None:
            # Find common ancestor for files in dataset
            reference_paths = list(
                map(lambda i: str(get_item_reference_path(*i)), self.iter_dicts())
            )
            common_path = os.path.commonpath(
                np.random.choice(
                    reference_paths, min(1000, len(reference_paths))
                ).tolist()
            )
            if not common_path:
                raise RuntimeError(
                    "Unable to find common ancestor for files in dataset. "
                    "Cannot set output_dir."
                )
            self.common_path = Path(common_path)

    def __len__(self) -> int:
        return len(self.data_ids)

    def __getitem__(self, index: int) -> AnnotatedAudio:
        data_id = self.data_ids[index]
        return self.get_item(data_id)

    def get_item(self, data_id: str) -> AnnotatedAudio:
        item = self.data[data_id]

        # Load the audio file and all vector files
        item_values: dict[str, Any] = {}
        if "audio" in item:
            item_values["audio"] = load_audio(
                item["audio"], self.sample_rate, normalize=self.normalize
            )
            item_values["audio_path"] = item["audio"]
        for vector_type in VectorType:
            if vector_type.value in item:
                item_values[vector_type] = load_vector_file(item[vector_type])

        if len(item_values) == 0:
            raise RuntimeError(
                f"Invalid empty item with id {data_id}: {item}. "
                "You must supply at least an audio or a vector file path for each item "
                "in an AudioDataset."
            )

        return AnnotatedAudio(
            id=data_id,
            sample_rate=self.sample_rate,
            speaker=item.get("speaker", None),
            **item_values,
        )

    def save_vector(
        self,
        id: str,
        vector: npt.NDArray,
        vector_type: VectorType,
        vector_extension: VectorExtension = VectorExtension.pkl,
    ):
        """
        Save a vector (embedding, binary atttributes) for a data item to disk and
        add it to this dataset.

        Args:
            id (str): the data item id.
            vector (npt.NDArray): the vector.
            vector_type (VectorType): the vector type.
            vector_extension (VectorExtension, optional): vector file extension.
                Defaults to VectorExtension.pkl.
        """
        # Get base path for audio or vector file already present in dataset
        item = self.data[id]
        basepath = get_item_reference_path(id, item)

        # If root_dir and output_dir are set, save vector file in output_dir with path
        # relative to root_dir
        if self.output_dir is not None and self.common_path is not None:
            basepath = self.output_dir / basepath.relative_to(self.common_path)
            basepath.parent.mkdir(exist_ok=True, parents=True)

        vector_file = vector_file_name(basepath, vector_type, vector_extension)
        if vector_extension is VectorExtension.txt:
            np.savetxt(vector_file, vector)
        else:
            with open(vector_file, "wb") as f:
                pickle.dump(vector, f, protocol=pickle.HIGHEST_PROTOCOL)
        item[vector_type] = vector_file

    def save_vectors(
        self,
        vectors: list[tuple[str, npt.NDArray]],
        vector_type: VectorType,
        vector_extension: VectorExtension = VectorExtension.pkl,
    ):
        """
        Save a list of vectors (embedding, binary attributes) to disk and add
        them to this dataset.

        Args:
            vectors (list[tuple[str, npt.NDArray]]): list of (id, vector).
            vector_type (VectorType): vector type (embedding, binary_attributes).
            vector_extension (VectorExtension, optional): vector file extension.
                Defaults to VectorExtension.pkl.
        """
        for id, vector in vectors:
            self.save_vector(id, vector, vector_type, vector_extension)

    def filter(self, func: Callable[[tuple[str, dict[str, Any]]], bool]) -> AudioDataset:
        filtered = list(filter(func, self.iter_dicts()))
        filtered_ids = [id for id, _ in filtered]
        filtered_data = dict(filtered)
        return AudioDataset(
            filtered_data,
            filtered_ids,
            sample_rate=self.sample_rate,
            sort=False,
            output_dir=self.output_dir,
        )

    def iter_dicts(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """
        Iterator on the underlying dict items. Useful when you need to iterate over
        the dataset without loading the audio signal or vector and simply want to access
        the data in the dataset's dicts.

        Yields:
            Iterator[tuple[str, dict[str, Any]]]: an iterator over the dataset's dicts
                of items
        """
        for id in self.data_ids:
            yield id, self.data[id]

    def to_csv(self, file: Path):
        """
        Export dataset to a csv file containing the id, speaker and file paths of
        each record in the dataset.

        Args:
            file (Path): the csv file path to export to.
        """
        records = [{"id": id, **data} for id, data in self.iter_dicts()]
        df = pd.DataFrame(records)
        df.to_csv(file, index=False)

    def __str__(self) -> str:
        embeddings = 0
        ba = 0
        speakers = []
        for id, data in self.iter_dicts():
            if "embedding" in data:
                embeddings += 1
            if "binary_attributes" in data:
                ba += 1
            if "speaker" in data:
                speakers.append(data["speaker"])
        classes = Counter(speakers)
        repr_str = (
            f"{self.__class__.__name__}: size={len(self.data_ids)}, sample_rate="
            f"{self.sample_rate}, embeddings={embeddings}, binary_attributes={ba}, "
            f"classes={len(classes)}."
        )
        return repr_str


def build_dataloader(
    dataset: AudioDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    rank: int,
    device: torch.device,
    collate_fn: Callable[[list[AnnotatedAudio]], Any],
    sampler: torch.utils.data.Sampler | None = None,
) -> DataLoader:
    """
    Create and return a DataLoader for the given dataset.

    Args:
        dataset (AudioDataset): dataset to load data from.
        batch_size (int): batch size for the dataloader.
        num_workers (int): number of worker threads for loading the data.
        shuffle (bool): whether to shuffle the dataset.
        rank (int): process rank for distributed processing. -1 for single-GPU training.
        device (torch.device): device used to process the data.
        collate_fn (Callable[[list[AnnotatedAudio]], Any]): collate function to collate
            batches of data.
        sampler (torch.utils.data.Sampler | None, optional): sampler for the dataloader.
            If sampler is None and rank != -1, a torch.utils.data.DistributedSampler will
            be used. Defaults to None.

    Returns:
        DataLoader: a dataloader on the dataset.
    """
    batch_size = max(min(batch_size, len(dataset)), 1)
    nd = torch.cuda.device_count()  # number of CUDA devices
    num_workers = min(os.cpu_count() // max(nd, 1), num_workers)  # type: ignore
    if device.type in {"cpu", "mps"}:
        num_workers = 0  # force workers to 0 when using CPU to improve loading time
    if sampler is None and rank != -1:
        sampler = distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader = DataLoader(
        dataset,
        shuffle=shuffle and sampler is None,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    return dataloader
