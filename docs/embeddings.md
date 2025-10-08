# Embedding extraction

Extraction of speaker embeddings is the first step towards producing binary-attribute based representations. The BA-LR toolkit provides a wrapper around [wespeaker](https://github.com/wenet-e2e/wespeaker)'s embedding extraction model.

## WespeakerModel

```python
class WespeakerModel(EmbeddingsModel):
    def __init__(
        self,
        model_repo: str = "Wespeaker/wespeaker-voxceleb-resnet34-LM",
        model_name: str = "avg_model",
        config_name: str = "config.yaml",
        model_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        features: FeaturesConfig = FeaturesConfig(),
    ):
```

**Parameters**:

* **model_repo**: the name of the [Wespeaker model repository](https://huggingface.co/Wespeaker) on huggingface to load the model from.
* **model_name**: the model weights file name within the repository (should usually be `avg_model.pt` or `avg_model`).
* **config_name**: the model config file name within the repository (should usually be `config.yaml`).
* **model_dir**: the local folder path where model files will be downloaded to. If None, the default will be `~/.cache`.
* **device**: the device to use the model on.
* **features**: config parameters for wespeaker's features extraction.

## Extract embeddings

```python
def extract_embeddings(
    self, dataset: AudioDataset, stream_save: bool, data_config: DataConfig
) -> list[tuple[str, npt.NDArray]]:
```

**Parameters**:

* **dataset**: the dataset to extract embeddings for.
* **stream_save**: if True, save the embeddings to disk as they are computed, returning an empty list. Otherwise, return a list of embedding and id tuples.
* **data_config**: config parameters for the `Dataloader` used with the dataset.

The `extract_embeddings` method can be called on the `WespeakerModel` class to extract embeddings for the audio files in the given dataset. Elements in the dataset without an `audio` waveform will be skipped.

## CLI

The BA-LR cli provides an `extract` command to extract embeddings for audio files in an `AudioDataset`.

**Parameters**:

* **input**: path to the dataset to process.
* **force**: whether to force extraction of samples that already have embeddings.
* **save_output**: whether to save embeddings to disk as they are extracted.
* **save_dir**: directory where the embeddings will be saved (sets the `output_dir` parameter on the `AudioDataset`).
* **audio_formats**: optional list of audio file extensions to load if `input` points to a directory of audio files.
* **device**: the device to use for embedding extraction.
* **overrides**: optional hydra config overrides.

!!! example

    ```bash
    balr extract resources/data/voxceleb2/metadata.csv
    ```
    will extract embedding vectors for all the audio files in the `voxceleb2` dataset.

!!! example

    ```bash
    balr extract --force --save-dir resources/data/voxceleb2-emb --device cuda resources/data/voxceleb2/metadata.csv embeddings.model.model_repo=Wespeaker/wespeaker-voxceleb-resnet293-LM embeddings.model.model_name=avg_model.pt
    ```

    will extract embedding vectors for all the audio files in the `voxceleb2` dataset, even those which already have embeddings (`--force`), saving those embeddings to a different directory (`--save-dir resources/data/voxceleb2-emb`), using a gpu for extraction (`--device cuda`) and using the `wespeaker-voxceleb-resnet34-LM` model.
