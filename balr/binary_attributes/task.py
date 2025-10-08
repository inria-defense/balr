import logging
from pathlib import Path
from typing import Annotated

import numpy.typing as npt
import typer
from hydra.utils import instantiate

from balr.binary_attributes.encoder import BinaryAttributeEncoder
from balr.config.configs import RunConfig, load_run_config
from balr.data.dataset import AudioDataset

app = typer.Typer()

LOGGER = logging.getLogger(__name__)


def extract_binary_attributes(
    dataset: AudioDataset,
    model: BinaryAttributeEncoder,
    config: RunConfig,
    force: bool,
) -> list[tuple[str, npt.NDArray]]:
    """
    Perform binarization of embedding vectors on dataset using
    BinaryAttributeEncoder provided.

    Args:
        dataset (AudioDataset): the dataset to perform embeddings extraction on.
        model (BinaryAttributeEncoder): the binary attribute encoder to use.
        config (RunConfig): task configuration.
        force (bool): whether to force binarization of samples that already have
            ba vectors.

    Returns:
        list[tuple[str, npt.NDArray]]: the computed binary attribute vectors.
    """
    if not force:
        # keep only items in dataset that do not already have ba vectors
        dataset = dataset.filter(
            lambda item: "binary_attributes" not in item[1]
            or item[1]["binary_attributes"] is None
        )

    if len(dataset) < 1:
        LOGGER.warning(
            "Dataset is empty or all samples already have BA vectors. Set `force` to "
            "True to force binarization of samples that already have BA vectors."
        )
        return []

    if config.save_dir is not None:
        dataset.set_output_dir(config.save_dir)

    stream_save = config.save_output
    binary_attributes = model.get_binary_attributes(
        dataset, stream_save, data_config=config.data
    )
    return binary_attributes


@app.command()
def binarize(
    input: Annotated[
        Path,
        typer.Argument(
            help="Path to input file or directory.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
        ),
    ],
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="Optional hydra config overrides."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(help="Force extraction of samples that already have embeddings."),
    ] = False,
    save_output: Annotated[bool, typer.Option(help="Save outputs.")] = True,
    save_dir: Annotated[
        Path | None,
        typer.Option(
            help="Save directory.",
            dir_okay=True,
        ),
    ] = None,
    exist_ok: Annotated[
        bool,
        typer.Option(
            help=(
                "Save results in save_dir even if it already exists. If false, "
                "save_dir will be incremented with a suffix if it already exists."
            )
        ),
    ] = False,
    audio_formats: Annotated[
        list[str],
        typer.Option(help="Extensions of audio files to load from input directory."),
    ] = ["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"],
    root_dir: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Root directory to use as prefix for relative paths in a csv metadata"
                " file."
            ),
            dir_okay=True,
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="device to use for predictions")] = "cpu",
    config_path: Annotated[
        str | None,
        typer.Option(help="hydra config path relative to the parent of the caller"),
    ] = None,
    config_name: Annotated[str, typer.Option(help="hydra config name")] = "config",
):
    """
    Run embedding binarization on inputs.

    Args:
        input (Path): Path to input file or directory.
        overrides (list[str] | None, optional): hydra config overrides. Defaults to None.
        force (bool, optional): Force extraction of samples that already have embeddings.
            Defaults to False.
        save_output (bool, optional): Save outputs. Defaults to True.
        save_dir (Path | None, optional): Save directory. If None, results are saved in
            dataset directory. Defaults to None.
        exist_ok (bool, optional): Save results in save_dir even if it already exists.
            If false, save_dir will be incremented with a suffix if it already exists.
            Defaults to False.
        audio_formats (list[str], optional): Extensions of audio files to load from input
            directory. Defaults to ["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"].
        root_dir (Path | None, optional): Root directory to use as prefix for relative
            paths in a csv metadata. If None, the csv file's parent directory is used.
            Defaults to None.
        device (str, optional): Device to use for predictions. Defaults to "cpu".
        config_path (str | None, optional): hydra config path relative to the parent of
            the caller. Defaults to None.
        config_name (str, optional): hydra config name. Defaults to "config".

    Example:
        To binarize embeddings for audio files in a given directory:

            $ balr binarize resources/data/voxceleb1

        To binarize embeddings for a single audio:
            $ balr binarize resources/data/voxceleb1/id10270+5r0dWxy17C8+00004.wav
    """
    # 1. load run configuration
    conf = load_run_config(
        input=input,
        overrides=overrides,
        save_output=save_output,
        save_dir=save_dir,
        exist_ok=exist_ok,
        audio_formats=audio_formats,
        device=device,
        config_path=config_path,
        config_name=config_name,
    )

    # 2. load dataset from directory or single file
    dataset = AudioDataset.from_path(conf.input, conf.audio_formats, root_dir=root_dir)

    # 3. load binary attribute encoder from configuration
    model = instantiate(conf.encoder.model, device=conf.device)

    # 4. binarize embeddings from dataset
    extract_binary_attributes(dataset, model, conf, force)


@app.command()
def train(
    train: Annotated[
        Path,
        typer.Argument(
            help="Path to train dataset.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
        ),
    ],
    val: Annotated[
        Path,
        typer.Argument(
            help="Path to val dataset.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
        ),
    ],
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="Optional hydra config overrides."),
    ] = None,
    save_dir: Annotated[
        Path | None,
        typer.Option(
            help="Save directory.",
            dir_okay=True,
        ),
    ] = None,
    exist_ok: Annotated[
        bool,
        typer.Option(
            help=(
                "Save results in save_dir even if it already exists. If false, "
                "save_dir will be incremented with a suffix if it already exists."
            )
        ),
    ] = False,
    audio_formats: Annotated[
        list[str],
        typer.Option(help="Extensions of audio files to load from input directory."),
    ] = ["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"],
    root_dir: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Root directory to use as prefix for relative paths in a csv metadata"
                " file."
            ),
            dir_okay=True,
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="device to use for predictions")] = "cpu",
    config_path: Annotated[
        str | None,
        typer.Option(help="hydra config path relative to the parent of the caller"),
    ] = None,
    config_name: Annotated[str, typer.Option(help="hydra config name")] = "config",
):
    """
    Run binary attribute encoder training on train and val datasets.

    Args:
        train (Path): Path to train dataset.
        val (Path): Path to validation dataset.
        overrides (list[str] | None, optional): hydra config overrides. Defaults to None.
        save_dir (Path | None, optional): Save directory. If None, defaults to
            'runs/train'. Defaults to None.
        exist_ok (bool, optional): Save results in save_dir even if it already exists.
            If false, save_dir will be incremented with a suffix if it already exists.
            Defaults to False.
        audio_formats (list[str], optional): Extensions of audio files to load from input
            directory. Defaults to ["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"].
        root_dir (Path | None, optional): Root directory to use as prefix for relative
            paths in a csv metadata. If None, the csv file's parent directory is used.
            Defaults to None.
        device (str, optional): Device to use for predictions. Defaults to "cpu".
        config_path (str | None, optional): hydra config path relative to the parent of
            the caller. Defaults to None.
        config_name (str, optional): hydra config name. Defaults to "config".

    Example:
        To run training on voxceleb2 sample dataset, using same dataset for validation,
        for 10 epochs:

            $ balr train resources/data/voxceleb2/metadata.csv \
                resources/data/voxceleb2/metadata.csv trainer.epochs=10

    """
    # 1. load run configuration
    conf = load_run_config(
        input=train,
        overrides=overrides,
        save_output=True,
        save_dir=save_dir,
        exist_ok=exist_ok,
        audio_formats=audio_formats,
        device=device,
        config_path=config_path,
        config_name=config_name,
    )

    # 2. load datasets
    train_dataset = AudioDataset.from_path(
        conf.input, conf.audio_formats, root_dir=root_dir
    )
    val_dataset = AudioDataset.from_path(val, conf.audio_formats, root_dir=root_dir)

    # 3. instantiate loss functions
    loss_funcs = [
        instantiate(f) for f in conf.trainer.losses.values() if f is not None  # type: ignore
    ]

    # 4. train binary attributes encoder
    trainer = instantiate(
        conf.trainer,
        train=train_dataset,
        val=val_dataset,
        data_config=conf.data,
        device=conf.device,
        save_dir=conf.save_dir,
        exist_ok=exist_ok,
        loss_funcs=loss_funcs,
    )

    trainer.train()
