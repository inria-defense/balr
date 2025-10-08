import logging
from pathlib import Path
from typing import Annotated

import numpy.typing as npt
import typer
from hydra.utils import instantiate

from balr.config.configs import RunConfig, load_run_config
from balr.data.dataset import AudioDataset
from balr.embeddings.embeddings import EmbeddingsModel

app = typer.Typer()

LOGGER = logging.getLogger(__name__)


def extract_embeddings(
    dataset: AudioDataset,
    model: EmbeddingsModel,
    config: RunConfig,
    force: bool,
) -> list[tuple[str, npt.NDArray]]:
    """
    Perform embeddings extraction on dataset using EmbeddingsModel provided.

    Args:
        dataset (AudioDataset): the dataset to perform embeddings extraction on.
        model (EmbeddingsModel): the EmbeddingsModel to use.
        config (RunConfig): task configuration.
        force (bool): whether to force extraction of samples that already have
            embeddings.

    Returns:
        list[tuple[str, npt.NDArray]]: the computed embeddings.
    """
    if not force:
        # keep only items in dataset that do not already have an embedding
        dataset = dataset.filter(
            lambda item: "embedding" not in item[1] or item[1]["embedding"] is None
        )

    if len(dataset) < 1:
        LOGGER.warning(
            "Dataset is empty or all samples already have embeddings. Set `force` to "
            "True to force embedding extraction of samples that already have embeddings."
        )
        return []

    if config.save_dir is not None:
        dataset.set_output_dir(config.save_dir)

    stream_save = config.save_output
    embeddings = model.extract_embeddings(dataset, stream_save, data_config=config.data)
    return embeddings


@app.command()
def extract(
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
    Run embedding extraction on inputs.

    Args:
        input (Path): Path to input file or directory.
        overrides (list[str] | None, optional): hydra config overrides. Defaults to None.
        force (bool, optional): Force extraction of samples that already have embeddings.
            Defaults to False.
        save_output (bool, optional): Save outputs. Defaults to True.
        save_dir (Path | None, optional): Save directory. If None, results are saved in
            dataset directory. Defaults to None.
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
        To extract embeddings from audio files in a given directory, and save embeddings
        as txt files:

            $ balr extract resources/data/voxceleb1

        To extract embeddings for a single audio and save embeddings as a txt file:
            $ balr extract resources/data/voxceleb1/id10270+5r0dWxy17C8+00004.wav
    """
    # 1. load run configuration
    conf = load_run_config(
        input=input,
        overrides=overrides,
        save_output=save_output,
        save_dir=save_dir,
        exist_ok=True,
        audio_formats=audio_formats,
        device=device,
        config_path=config_path,
        config_name=config_name,
    )

    # 2. load dataset from directory or single file
    dataset = AudioDataset.from_path(conf.input, conf.audio_formats, root_dir=root_dir)

    # 3. load embeddings model from configuration
    model = instantiate(conf.embeddings.model, device=conf.device)

    # 4. extract embeddings from dataset
    extract_embeddings(dataset, model, conf, force)
