from collections.abc import Iterable
from enum import Enum

import numpy as np
import numpy.typing as npt
from pandas import DataFrame, Series

from balr.data.dataset import AudioDataset


class TrialsColumn(str, Enum):
    enrollment = "enrollment"
    test = "test"


def get_trials_column(
    trials: DataFrame, column: TrialsColumn = TrialsColumn.enrollment
) -> Series:
    """
    Get the column from a trials list dataframe. By default, look for the column by name.
    If there is no column with the required name, assume that enrollment column is the
    first one, and test column is the second one.

    Args:
        trials (DataFrame): the trials dataframe, containing enrollment and test ids.
        column (TrialsColumn, optional): the column to get.
            Defaults to TrialsColumn.enrollment.

    Returns:
        Series: the requested column from the dataframe.
    """
    columns = list(map(lambda x: x.lower(), trials.columns.to_list()))
    if column.value in columns:
        idx = columns.index(column.value)
    elif column is TrialsColumn.enrollment:
        idx = 0
    else:
        idx = 1
    return trials.iloc[:, idx]


def get_column_labels(dataset: AudioDataset, ids: Iterable[list[str]]) -> list[str]:
    """
    Get labels (speakers) for a list of ids from the dataset.

    Args:
        dataset (AudioDataset): the dataset.
        ids (Iterable[list[str]]): list of ids, each row being a list of one or more
            sample ids from the dataset.

    Raises:
        ValueError: if an element does not have a speaker label in the dataset,
            or if elements from the same row have different speakers.

    Returns:
        list[str]: the list of speakers corresponding to each row of ids.
    """
    labels: list[str] = []
    for element in ids:
        element_labels: set[str] = set()
        for id in element:
            sample = dataset.data[id]
            if "speaker" not in sample or sample["speaker"] is None:
                raise ValueError(f"Speaker for {id} sample is None.")
            element_labels.add(sample["speaker"])
        if len(element_labels) > 1:
            raise ValueError(
                f"Enrollment or test trial with different labels (speaker): {element}"
            )
        labels.append(element_labels.pop())
    return labels


def get_trial_labels(dataset: AudioDataset, trials: DataFrame) -> npt.NDArray[np.int_]:
    """
    Get labels array from a trials list. Labels are 0 if speakers are different for
    enrollment and test samples, and 1 otherwise.

    Args:
        dataset (AudioDataset): the test dataset.
        trials (DataFrame): the trials list.

    Returns:
        npt.NDArray[np.int_]: the labels array.
    """
    enroll_ids = get_trials_column(trials, TrialsColumn.enrollment).str.split(",")
    test_ids = get_trials_column(trials, TrialsColumn.test).str.split(",")

    enroll_labels = np.array(get_column_labels(dataset, enroll_ids), dtype=str)
    test_labels = np.array(get_column_labels(dataset, test_ids), dtype=str)

    labels = (enroll_labels == test_labels).astype(int)
    return labels


def get_positive_negative_scores(
    scores: npt.NDArray[np.float_], labels: npt.NDArray[np.int_]
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Split the scores array into positive and negative scores. Positive scores
    correspond to scores where labels > 0, and negative scores to those where labels == 0.

    Args:
        scores (npt.NDArray[np.float_]): the scores array.
        labels (npt.NDArray[np.int_]): the labels array.

    Returns:
        tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]: the positive and
            negative scores.
    """
    positive_scores = scores[labels.astype(bool)]
    negative_scores = scores[~labels.astype(bool)]
    return positive_scores, negative_scores
