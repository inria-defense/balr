import json
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from hydra.utils import instantiate

from balr.config.configs import DataConfig, load_run_config
from balr.data.dataset import AudioDataset
from balr.data.stats import get_speaker_statistics
from balr.evaluation.metrics import compute_metrics
from balr.scoring.scoring import Scorer
from balr.scoring.utils import TrialsColumn, get_trials_column
from balr.utils import setup_save_dir

app = typer.Typer()

LOGGER = logging.getLogger(__name__)


def train_scorer(
    scorer: Scorer, dataset: AudioDataset, n_iterations: int, data_config: DataConfig
):
    """
    Train scorer by
        1. getting speaker activation statistics from dataset,
        2. call scorer.fit with tensor of speaker activation statistics,
        3. save scorer weights to disk.

    Args:
        scorer (Scorer): the scorer to train.
        dataset (AudioDataset): the training dataset.
        n_iterations (int): number of iterations when using an estimation algorithm.
        data_config (DataConfig): dataloader config.
    """
    referential = get_speaker_statistics(dataset, data_config)
    scorer.fit(referential, n_iterations=n_iterations)
    scorer.save_model()


def get_activation_tensors(dataset: AudioDataset, ids: list[list[str]]) -> torch.Tensor:
    """
    Get the binary attribute vectors for a list of ids from the given dataset.

    Args:
        dataset (AudioDataset): the dataset.
        ids (list[list[str]]): list of ids.

    Raises:
        ValueError: if an element does not have binary attributes in the dataset.

    Returns:
        torch.Tensor: the stacked binary attribute tensors.
    """
    activations = []
    for element in ids:
        element_activations = []
        for id in element:
            ba = dataset.get_item(id).binary_attributes
            if ba is None:
                raise ValueError(f"Binary Attributes Vector for {id} sample is None.")
            element_activations.append(ba)

        if len(element_activations) == 1:
            activations.append(element_activations[0])
        else:
            activations.append(torch.stack(element_activations).sum(0))

    return torch.stack(activations)


def score(
    scorer: Scorer,
    dataset: AudioDataset,
    trials: pd.DataFrame,
    save_dir: Path | None,
    exist_ok: bool,
    sep: str,
):
    """
    Run scoring on the provided list of trials (pairs of enrollment and test
    samples).

    Args:
        scorer (Scorer): the scorer.
        dataset (AudioDataset): the test dataset. It needs to contain all the items
            listed in the trials list.
        trials (pd.DataFrame): trials list. A DataFrame with two columns (enrollments
            and tests) containing comma separated lists of ids to score.
        save_dir (Path | None): directory where scores will be save.
        exist_ok (bool): save results in save_dir even if it already exists.
        sep (str): separator char for the output csv file.
    """
    enroll_ids = get_trials_column(trials, TrialsColumn.enrollment).str.split(",")
    test_ids = get_trials_column(trials, TrialsColumn.test).str.split(",")

    enrollment_activations = get_activation_tensors(dataset, enroll_ids.to_list())
    test_activations = get_activation_tensors(dataset, test_ids.to_list())

    scores = scorer(enrollment_activations, test_activations)
    trials["scores"] = scores.detach().cpu().numpy()
    save_dir = setup_save_dir(save_dir, exist_ok=exist_ok, default_path="runs/score")
    score_file = save_dir / "scores.csv"
    trials.to_csv(score_file, index=False, float_format="%.4f", sep=sep)
    LOGGER.info(f"Scores saved to {score_file}")

    metrics = compute_metrics(dataset, trials)
    metrics_str = [f"{key}: {value:.4f}" for key, value in metrics.items()]
    LOGGER.info(
        "<==================== Evaluation metrics ====================>\n"
        + ", ".join(metrics_str)
        + "\n<============================================================>"
    )
    metrics_file = save_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)


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
    Run scorer training on a training dataset.

    Args:
        train (Path): Path to train dataset.
        overrides (list[str] | None, optional): hydra config overrides. Defaults to None.
        save_dir (Path | None, optional): Save directory. If None, defaults to
            'runs/scorer/train'. Defaults to None.
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
        To run training on voxceleb2 sample dataset:

            $ balr score train resources/data/voxceleb2/metadata.csv

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

    # 2. load dataset
    dataset = AudioDataset.from_path(conf.input, conf.audio_formats, root_dir=root_dir)

    # 3. instantiate scorer
    scorer = instantiate(
        conf.scorer, save_dir=conf.save_dir, exist_ok=exist_ok, device=conf.device
    )

    # 4. train scorer on dataset
    train_scorer(scorer, dataset, conf.scorer.n_iterations, conf.data)


@app.command()
def test(
    test: Annotated[
        Path,
        typer.Argument(
            help="Path to test dataset.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
        ),
    ],
    trials: Annotated[
        Path,
        typer.Argument(
            help="Path to trials list.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="Optional hydra config overrides."),
    ] = None,
    sep: Annotated[str, typer.Option(help="Separator char for trials list.")] = "\t",
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
    Run scorer on a test dataset with a list of trial (enrollment and test pairs).

    Args:
        test (Path): Path to test dataset.
        trials (Path): Path to trials list. Should be a csv file containing ids for
            enrollment and test pairs.
        overrides (list[str] | None, optional): hydra config overrides. Defaults to None.
        sep (str, optional): Separator char for trials list. Defaults to '\t'.
        save_dir (Path | None, optional): Directory where scores are save.
            If None, defaults to 'runs/score'. Defaults to None.
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
        To run scoring on voxceleb2 sample dataset:

            $ balr score test resources/data/voxceleb2/test.csv \
                resources/data/voxceleb2/trials.csv \
                scorer.checkpoint_path=runs/scorer/train/scorer.pt

    """
    # 1. load run configuration
    conf = load_run_config(
        input=test,
        overrides=overrides,
        save_output=True,
        save_dir=save_dir,
        exist_ok=exist_ok,
        audio_formats=audio_formats,
        device=device,
        config_path=config_path,
        config_name=config_name,
    )

    # 2. load dataset
    dataset = AudioDataset.from_path(conf.input, conf.audio_formats, root_dir=root_dir)

    # 3. load trials list
    trials_list = pd.read_csv(trials, sep=sep)

    # 4. instantiate scorer
    scorer = instantiate(
        conf.scorer, save_dir=conf.save_dir, exist_ok=exist_ok, device=conf.device
    )

    # 5. run scoring on trials from dataset
    score(scorer, dataset, trials_list, conf.save_dir, conf.exist_ok, sep)
