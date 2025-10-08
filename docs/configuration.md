# Configuration

## CLI options

The BA-LR toolkit provides a command line interface (CLI) for running the most common tasks decribed in this documentation. The commands in the CLI accept command line options prefixed with `--` and are specified when the command is run with the `--help` flag.

To see the list of available commands, run

```bash
balr --help

Usage: balr [OPTIONS] COMMAND [ARGS]...

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                        │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                 │
│ --help                        Show this message and exit.                                                                                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ extract    Run embedding extraction on inputs.                                                                                                 │
│ binarize   Run embedding binarization on inputs.                                                                                               │
│ train      Run binary attribute encoder training on train and val datasets.                                                                    │
│ score      Scoring related commands.                                                                                                           │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

To see the options available for a specific command, run that command with the `--help` flag

```bash
balr extract --help

Usage: balr extract [OPTIONS] INPUT [OVERRIDES]...

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    input          PATH            Path to input file or directory. [default: None] [required]                                                │
│      overrides      [OVERRIDES]...  Optional hydra config overrides. [default: None]                                                           │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --force            --no-force                Force extraction of samples that already have embeddings. [default: no-force]                     │
│ --save-output      --no-save-output          Save outputs. [default: save-output]                                                              │
│ --save-dir                             PATH  Save directory. [default: None]                                                                   │
│ --audio-formats                        TEXT  Extensions of audio files to load from input directory.                                           │
│                                              [default: flac, mp3, m4a, ogg, opus, wav, wma]                                                    │
│ --device                               TEXT  device to use for predictions [default: cpu]                                                      │
│ --config-path                          TEXT  hydra config path relative to the parent of the caller [default: None]                            │
│ --config-name                          TEXT  hydra config name [default: config]                                                               │
│ --help                                       Show this message and exit.                                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Most command line options let you set parameters related to how the task will run, such as setting the device (i.e. `--device cuda`) or specifying where the results will be saved (`--save-dir /path/tooutput/dir`).

## Hydra configuration

The CLI commands also accept optional [hydra configuration](https://hydra.cc/docs/tutorials/structured_config/intro/) overrides for the various models used by the tasks. The structured config dataclasses used by the BA-LR toolkit are available in the `balr.config.configs.py` module and described below.

### Data loading configuration

The `data` configuration class lets you specify parameters related to how the dataset will be loaded and sampled.

```python
@dataclass
class DataConfig:
    batch_size: int = 64
    num_workers: int = 1
    shuffle: bool = True
    sampler: str | None = "balr.samplers.nxm_samplers.RandomNxMSampler"
    N_classes_per_batch: int = 8
    M_samples_per_class: int = 8
```

**Parameters**:

* **batch_size**: the `batch_size` parameter for the [DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).
* **num_workers**: the number of workers used by the [DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html). Only used for tasks running on GPUs. When the device is `cpu`, the number of workers is forced to 0.
* **shuffle**: the `shuffle` parameters for the [DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) and [Sampler](./samplers.md) classes.
* **sampler**: the [Sampler](./samplers.md) class to use with the [DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).
* **N_classes_per_batch**: `N` parameter when using an [NxMSampler](./samplers.md) class
* **M_samples_per_class**: `M` parameter when using an [NxMSampler](./samplers.md) class

!!! example

    ```bash
    balr train --device cuda resources/data/voxceleb2/train.csv resources/data/voxceleb2/test.csv data.num_workers=4 data.shuffle=False data.sampler=balr.samplers.nxm_samplers.ExhaustiveNxMSampler data.N_classes_per_batch=32 data.M_samples_per_class=10 data.batch_size=320
    ```

    This example command will train a binary-attribute AutoEncoder on a GPU (`--device cuda`) with 4 workers for the DataLoader (`data.num_workers=4`), using an `ExhaustiveNxMSampler` with a batch size of `N (32) x M (10) = 320`.

!!! note

    Note that when using an [NxMSampler](./samplers.md), the `batch_size` parameter must be set equal to `N x M`.

### Embeddings configuration

### Binary-attribute encoder configuration

### Binary-attribute encoder trainer configuration

### Scorer configuration
