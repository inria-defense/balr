# Binary-Attribute Encoder training

The BA-LR toolkit provides a `BinaryAttributeEncoderTrainer` class to train an AutoEncoder for binary-attribute representation extraction from embedding vectors.

## BinaryAttributeEncoderTrainer

```python
class BinaryAttributeEncoderTrainer:
    def __init__(
        self,
        train: AudioDataset,
        val: AudioDataset,
        data_config: DataConfig,
        loss_funcs: list[BaseAutoEncoderLoss],
        input_dim: int = 256,
        internal_dim: int = 512,
        learning_rate: float = 0.001,
        epochs: int = 100,
        seed: int = 1234,
        save_dir: Path | None = None,
        save_period: int = 0,
        log_period: int = 2,
        val_period: int = 10,
        device: str | torch.device = "cpu",
        **kwargs,
    ):
```

**Parameters**:

* **train**: the training dataset.
* **val**: the validation dataset.
* **data_config**: config parameters for the `Dataloader` used with the datasets.
* **loss_funcs**: a list of [loss functions](./losses.md) to use during training and validation.
* **input_dim**: the input dimension for the AutoEncoder (i.e. the embedding dimension).
* **internal_dim**: the internal or output dimension for the AutoEncoder (i.e. the dimension of the binary-attribute representation).
* **learning_rate**: the learning rate.
* **epochs**: the number of epochs for training.
* **seed**: the seed for random functions.
* **save_dir**: directory where the training output (logs, metrics and model checkpoints) are saved.
* **save_period**: save model checkpoint every x epochs.
* **log_period**: logs metrics every x epochs.
* **val_period**: run validation every x epochs.
* **device**: the device to use the model on.

## Training

Once the `BinaryAttributeEncoderTrainer` class has been initialized with the proper parameters, the `train` method will run training on the training dataset for the set number of epochs.

## CLI

The BA-LR cli provides a `train` command to train a `BinaryAttributeEncoder` using the `BinaryAttributeEncoderTrainer`.

**Parameters**:

* **train**: the path to the training dataset.
* **val**: the path to the validation dataset.
* **save_dir**: directory where the training output (logs, metrics and model checkpoints) are saved. By default, results will be saved to `./runs/trainX`, X being incremented as needed (`train2`, `train3`, etc. on successive runs).
* **device**: the device to use the model on.
* **overrides**: optional hydra config overrides.

!!! warning

    Both the training and validation datasets must provide embeddings for all their samples. But since only embeddings are needed for training, you can run training on datasets that *only provide embeddings* (i.e. without audio files).

!!! example

    ```bash
    balr train resources/data/voxceleb2/train.csv resources/data/voxceleb2/test.csv
    ```

    will train a `BinaryAttributeAutoEncoder` model on the `voxceleb2/train.csv` dataset and use the `voxceleb2/test.csv` dataset for validation. Results will be saved by default to the `./runs/train` directory. Training will run by default on the `cpu` device.


!!! example

    ```bash
    balr train --save-dir training_output --device cuda resources/data/voxceleb2/train.csv resources/data/voxceleb2/test.csv trainer.epochs=10 'trainer.losses=[mse, arcface]'
    ```

    This more complex command will run training on the same datasets, but specifies the device (`cuda`) and the output directory, as well as modifies the trainer's config for `epochs` and `losses` parameters using hydra overrides.


## Distributed training

It is possible to run training on multiple GPUs with pytorch's [torchrun](https://pytorch.org/docs/stable/elastic/run.html) launcher script.

!!! note

    When using `torchrun`, you must call the cli's `balr/cli/main.py` module instead of the `balr` command.

!!! example

    ```bash
    torchrun --nproc_per_node 2 balr/cli/main.py train resources/data/voxceleb2/train.csv resources/data/voxceleb2/test.csv --device cuda:0,1
    ```

    will run training on two GPUs. The [samplers](./samplers.md) used with the dataloaders must support distributed sampling.
