# BA-LR: Binary-Attribute based Likelihood Ratio estimation for explainable speaker recognition

## Documentation

**The documentation for BALR is available [here](https://inria-defense.github.io/balr/).**

## Install

BALR requires a recent version of python: ![python_version](https://img.shields.io/badge/Python-%3E=3.12-blue).

### Install from github

Clone the repository and install the project in your python environment, either using `pip`

```console
git clone https://github.com/inria-defense/balr.git
cd balr
pip install --editable .
```

or [uv](https://docs.astral.sh/uv/)

```console
git clone https://github.com/inria-defense/balr.git
cd balr
uv sync
```

## Usage

### CLI

When you install BALR in a virtual environment, it creates a CLI script called `balr`. Run

```console
balr --help
```

to see the various commands available (or take a look at the [documentation](https://inria-defense.github.io/balr/) for examples).

#### Embedding extraction

To perform embedding extraction on a set of audio files, run

```console
balr extract /path/to/dataset
```

That command will look for all audio files in the directory, and extract embeddings using the default embedding model ([Wespeaker](https://github.com/wenet-e2e/wespeaker)). Embeddings for each audio file will be saved as numpy arrays in `.txt` files matching the names of the audio file they correspond to.

#### Binarization of embeddings

To perform binarization of embeddings for a set of audio files, run

```console
balr binarize /path/to/dataset
```

That command will load embedding files in the dataset and perform binary attribute encoding using the default Binary Attribute Encoder (`AutoEncoder` using the saved checkpoint in `resources/model/binary_encoder`). BA vectors for each audio file will be saved as numpy arrays in `.txt` files matching the names of the embedding file they correspond to.

#### Training of the Binary Attribute Encoder

To train the Binary Attribute Encoder on a dataset, run

```console
balr train /path/to/train_dataset /path/to/validation_dataset
```

Parameters for the training process, such as learning rate or number of epochs, are defined in the [BATrainerConfig](balr/config/configs.py) class. You can override these parameters using [hydra](https://hydra.cc/docs/intro/) syntax. For example, run

```console
balr train /path/to/train_dataset /path/to/validation_dataset trainer.epochs=50 trainer.learning_rate=0.002
```

to change the `learning_rate` and `epochs` parameters.

##### Training on multiple gpus

Training on multiple gpus is possible using [torchrun](https://pytorch.org/docs/stable/elastic/run.html#launcher-api). When using `torchrun`, you must call the cli's [main](balr/cli/main.py) module instead of the `balr` command. For example, to run training on two gpus, run

```console
torchrun --nproc_per_node 2 balr/cli/main.py train /path/to/train_dataset /path/to/validation_dataset --device cuda:0,1
```

## Development

We recommend using [uv](https://docs.astral.sh/uv/) as a package and project manager when developing. Install `uv` then run `uv sync` to install the project and its dependencies.

### Format

Please format your code before contributing. Format your code with [black](https://github.com/psf/black): `uv run black --config ./pyproject.toml .`.

We recommand using [pre-commit](https://pre-commit.com/) to make sure your code is formatted before each commit. Run `uv run pre-commit install` from the root directory to install pre-commit hooks.
