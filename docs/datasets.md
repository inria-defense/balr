# Datasets

The BA-LR toolkit provides a custom dataset class (which extends the [`torch.utils.data.Dataset`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) class) to manage audio files and their vector representations.

## AudioDataset

The `AudioDataset` represents a dataset of `AnnotatedAudio` items. Each item in the dataset is identified by an id and provides information for an audio file, its labels (speaker id) and optional computed attributes, such as an embedding vector, or binary attribute vectors.

The dataset takes as input a dict of dicts as the collection of data points to read. The top level keys are data point IDs. Each data point dict should have the same keys, corresponding to different files in that data point.

For example the input data could look like this

```python
data = {
    "spk1utt1": {
        "audio": "/path/to/spk1utt1.wav",
        "embedding": "/path/to/spk1utt1_emb.txt",
        "binary_attributes": "/path/to/spk1utt1_ba.txt",
    },
    "spk1utt2": {
        "wav_file": "/path/to/spk1utt2.wav",
        "embedding": "/path/to/spk1utt2_emb.txt",
        "binary_attributes": "/path/to/spk1utt2_ba.txt",
    }
}
```

!!! note
    The files are only loaded in memory when the element they belong to is accessed.


## AnnotatedAudio

Elements of an `AudioDataset` can be accessed by their index. They are returned as an `AnnotatedAudio` object containing the element's id, label, and tensors corresponding to the audio signal, and embedding or binary attribute representations.

```python
class AnnotatedAudio:
    id: str
    sample_rate: int
    audio_path: Path | None = None
    audio: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    binary_attributes: torch.Tensor | None = None
    speaker: str | int | None = None
```

!!! warning
    At least one of `audio`, `embedding` or `binary_attributes` must be provided for each element in an `AudioDataset`.

## AudioDataset.from_path

The `AudioDataset` class offers a convenience method to create a dataset from a file, a directory of files, or a csv file (containing metadata information on the files in the dataset).

```python
@classmethod
def from_path(
    cls,
    path: str | Path,
    audio_formats: list[str] = ["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"],
    sample_rate: int = 16_000,
) -> AudioDataset:
```

**Parameters**:

* **path**: a path pointing to a file or directory.
* **audio_formats**: audio file extensions to look for if `path` is a directory.
* **sample_rate**: the target sample rate at which audio files will be loaded or resampled to if needed.

!!!warning
    If `path` points to a csv file, the csv file must contain at least one column pointing to `audio`, `embedding` or `binary_attributes` file paths, plus other optional columns such as speaker or id. If the id column is not specified, the name of the audio files will be used as ids instead (the `audio` column must be present).

For example, if the csv file looks like this

```csv
audio,speaker
id07417/00028.wav,id07417
id03184/00022.wav,id03184
id03184/00053.wav,id03184
id04961/00169.wav,id04961
id04961/00289.wav,id04961
id01184/00133.wav,id01184
id06261/00159.wav,id06261
id06261/00233.wav,id06261
id06261/00190.wav,id06261
id07531/00142.wav,id07531
```

Calling `AudioDataset.from_path` with the path of this csv file will return a Dataset of 10 samples, each sample having an audio file and being labeled with the speaker id from the csv file.

!!! warning

    The csv file describing a dataset can contain absolute or relative paths for the `audio`, `embedding` or `binary_attributes` file paths. **If these columns contain relative paths, they must be relative to the directory that contains the csv file.** In the exampe above, the various speaker directories `id07417`, `id03184`, etc. must be in the same directory as the csv file.

## Vector files

By default, the `AudioDataset` class will try to look for and load two types of vector files for each sample in the dataset:

* embedding files with the suffix `_emb`
* binary attribute files with the suffix `_ba`

Vector files should have the same name and path as the audio file to which they correspond, with the correct suffix appended. For exemple, if the directory structure looks like

    spk1utt1.wav        # The speaker audio file.
    spk1utt1_emb.txt    # The speaker embedding vector
    spk1utt1_ba.txt     # The speaker binary-attribute vector

Then the `AudioDataset` will automatically load and associate the embedding and binary-attribute files to the audio file with the same name.

The `AudioDataset` class supports loading vector files in **numpy text format** (with `.txt` extension) or as **pickled numpy arrays** (with `.pkl` extension).

## Loading a dataset of vector files only

In some cases, you might want to create a dataset of only vector files (embeddings or binary-attributes) without a reference to the audio file itself. This can be done by creating a csv file with a column pointing to the path of the vector file for each sample in the dataset. **When no `audio` column is provided, an `id` column is required for each sample as well as either an `embedding` or a `binary_attributes` column.**

For example, with a csv file like

```csv
id,binary_attributes,speaker
id07417/00028,id07417/00028_ba.txt,id07417
id03184/00022,id03184/00022_ba.txt,id03184
id03184/00053,id03184/00053_ba.txt,id03184
id04961/00169,id04961/00169_ba.txt,id04961
id04961/00289,id04961/00289_ba.txt,id04961
id01184/00133,id01184/00133_ba.txt,id01184
id06261/00159,id06261/00159_ba.txt,id06261
id06261/00233,id06261/00233_ba.txt,id06261
id06261/00190,id06261/00190_ba.txt,id06261
id07531/00142,id07531/00142_ba.txt,id07531
```

Calling `AudioDataset.from_path` with the path of this csv file will return a Dataset of 10 samples, each sample having an id, speaker, and binary_attributes vector.

## Saving vector files

The `save_vectors` method on an `AudioDataset` allows saving vectors for elements of the dataset. It will both save the given vectors to disk, and add them to the corresponding elements of the dataset.

```python
def save_vectors(self, vectors: list[tuple[str, npt.NDArray]], vector_type: VectorType):
    """
    Save a list of vectors (embedding, binary attributes) to disk and add
    them to this dataset.
    """
```

**Parameters**:

* **vectors**: a list of (id, vector) tuples to save.
* **vector_type**: the type of vector, either `embedding` or `binary_attributes`.

!!! tip
    By default, the vector files will be saved in the same directory as the audio file for the given element. If you want to save the vector files in a different directory, for instance if you do not have the permissions to write to the directory where the audio files are located, you can set the `output_dir` attribute on the `AudioDataset` with the `AudioDataset.set_output_dir` method. If `output_dir` is set on the `AudioDataset`, calling `AudioDataset.save_vectors` will instead save the vector files in the `output_dir` with the same directory structure as the audio files.
