# Binary-Attribute representation extraction

The BA-LR toolkit provides an `AutoEncoder` model architecture to extract binary-attribute representations. The `BinaryAttributeAutoEncoder` class must use a trained model checkpoint to perform inference of binary-attribute representations. The training of the `BinaryAttributeAutoEncoder` can be done with the [trainer](./trainer.md) module provided by the BA-LR toolkit.

## BinaryAttributeAutoEncoder

```python
class BinaryAttributeAutoEncoder(BinaryAttributeEncoder):
    def __init__(
        self,
        checkpoint_path: str | Path,
        input_dim: int = 256,
        internal_dim: int = 512,
        device: str | torch.device = "cpu",
    ):
```

**Parameters**:

* **checkpoint_path**: path to a trained model checkpoint. By default, uses the one provided in `resources/models/BAE/BAE_mse.pt`.
* **input_dim**: the input dimension for the AutoEncoder (i.e. the embedding dimension).
* **internal_dim**: the internal or output dimension for the AutoEncoder (i.e. the dimension of the binary-attribute representation).
* **device**: the device to use the model on.

## Extract binary-attribute representations

```python
def get_binary_attributes(
    self, dataset: AudioDataset, stream_save: bool, data_config: DataConfig
) -> list[tuple[str, npt.NDArray]]:
```

**Parameters**:

* **dataset**: the dataset to extract binary-attribute vectors for.
* **stream_save**: if True, save the binary-attribute vectors to disk as they are computed, returning an empty list. Otherwise, return a list of id and binary-attribute vector  tuples.
* **data_config**: config parameters for the `Dataloader` used with the dataset.

The `get_binary_attributes` method can be called on the `BinaryAttributeAutoEncoder` class to extract binary-attribute representations for the embedding files in the given dataset. Elements in the dataset without an `embedding` vector will be skipped.

## CLI

The BA-LR cli provides a `binarize` command to extract binary-attribute representations from embedding vectors in an `AudioDataset`.

**Parameters**:

* **input**: path to the dataset to process.
* **force**: whether to force extraction of samples that already have binary-attribute vectors.
* **save_output**: whether to save binary-attribute vectors to disk as they are extracted.
* **save_dir**: directory where the binary-attribute vectors will be saved (sets the `output_dir` parameter on the `AudioDataset`).
* **audio_formats**: optional list of audio file extensions to load if `input` points to a directory of audio files.
* **device**: the device to use for binarization.
* **overrides**: optional hydra config overrides.

!!! example

    ```bash
    balr binarize resources/data/voxceleb2/metadata.csv
    ```
    will extract binary-attribute vectors for all the embedding vectors present in the `voxceleb2` dataset.

!!! example

    ```bash
    balr binarize --force --save-dir resources/data/voxceleb2-ba --device cuda resources/data/voxceleb2/metadata.csv encoder.model.checkpoint_path=runs/train/best.pt
    ```

    will extract binary-attribute vectors for all the embedding vectors present in the `voxceleb2` dataset, even those which already have binary-attribute vectors (`--force`), saving these vectors to a different directory (`--save-dir resources/data/voxceleb2-ba`), using a gpu for binarization (`--device cuda`) and using a custom trained model checkpoint for the `AutoEncoder` (`encoder.model.checkpoint_path=runs/train/best.pt`).
