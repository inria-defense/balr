from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from pandas import DataFrame
from speechbrain.utils import metric_stats
from speechbrain.utils.metric_stats import BinaryMetricStats

from balr.data.dataset import AudioDataset
from balr.scoring.utils import (
    TrialsColumn,
    get_positive_negative_scores,
    get_trial_labels,
    get_trials_column,
)


def compute_metrics(dataset: AudioDataset, trials: DataFrame) -> dict[str, float]:
    """
    Compute a dictionnary of metrics from a trials dataframe containing the scores.

    Args:
        dataset (AudioDataset): the test dataset.
        trials (DataFrame): the trials dataframe containing enrollment and test ids as
            well as scores computed by a scorer.

    Raises:
        ValueError: if trials dataframe does not have a `scores` column.

    Returns:
        dict[str, float]: a dictionnary of metrics.
    """
    if "scores" not in trials:
        raise ValueError("Trials dataframe does not have a `scores` column.")

    scores = trials["scores"].to_numpy()
    labels = get_trial_labels(dataset, trials)
    enroll_ids = get_trials_column(trials, TrialsColumn.enrollment).astype(str)
    test_ids = get_trials_column(trials, TrialsColumn.test).astype(str)
    ids = (enroll_ids + "/" + test_ids).to_list()

    metrics = BMS(ids, scores, labels)

    positive_scores, negative_scores = get_positive_negative_scores(scores, labels)
    eer, thres = EER(positive_scores, negative_scores)
    cllr = CLLR(positive_scores, negative_scores)
    metrics.update({"EER": eer, "EER_thres": thres, "CLLR": cllr})
    return metrics


def BMS(
    ids: list[str],
    scores: npt.NDArray[np.float_],
    labels: npt.NDArray[np.int_],
    positive_label: Any = 1,
    threshold: float | None = None,
    max_samples: int | None = None,
    beta: float = 1.0,
) -> dict[str, float]:
    """
    Compute statistics using a full set of scores.

    Full set of fields:
        - TP - True Positive
        - TN - True Negative
        - FP - False Positive
        - FN - False Negative
        - FAR - False Acceptance Rate
        - FRR - False Rejection Rate
        - DER - Detection Error Rate (EER if no threshold passed)
        - threshold - threshold (EER threshold if no threshold passed)
        - precision - Precision (positive predictive value)
        - recall - Recall (sensitivity)
        - F-score - Balance of precision and recall (equal if beta=1)
        - MCC - Matthews Correlation Coefficient

    Args:
        ids (list[str]): list of ids for the trials.
        scores (npt.NDArray[np.float_]): the scores for each trial,
            of dimension (nb_trials,).
        labels (npt.NDArray[np.int_]): the labels for each trial,
            of dimension (nb_trials,).
        positive_label (Any, optional): positive label value. Defaults to 1.
        threshold (float | None, optional): threshold for selecting predictions.
            If no threshold is provided, equal error rate is used. Defaults to None.
        max_samples (int | None, optional): How many samples to keep for positive/negative
            scores. If no max_samples is provided, all scores are kept.
            Only effective when threshold is None. Defaults to None.
        beta (float, optional): How much to weight precision vs recall in F-score. Default
            of 1. is equal weight, while higher values weight recall
            higher, and lower values weight precision higher. Defaults to 1.0.

    Returns:
        dict[str, float]: dictionnary of metric values.
    """
    bms = BinaryMetricStats(positive_label=positive_label)
    bms.append(ids, torch.tensor(scores), torch.tensor(labels))

    return bms.summarize(
        field=None, threshold=threshold, max_samples=max_samples, beta=beta  # type: ignore
    )


def EER(
    positive_scores: npt.NDArray[np.float_], negative_scores: npt.NDArray[np.float_]
) -> tuple[float, float]:
    """
    Computes the EER (and its threshold).

    Args:
        positive_scores (npt.NDArray[np.float_]): The scores from entries of the same
            class.
        negative_scores (npt.NDArray[np.float_]): The scores from entries of different
            classes.

    Returns:
        tuple[float, float]: The EER score and its corresponding threshold.

    Example:
        >>> positive_scores = np.array([0.6, 0.7, 0.8, 0.5])
        >>> negative_scores = np.array([0.4, 0.3, 0.2, 0.1])
        >>> val_eer, threshold = EER(positive_scores, negative_scores)
        >>> val_eer
            0.0
    """
    return metric_stats.EER(torch.tensor(positive_scores), torch.tensor(negative_scores))


def negative_log_sigmoid(lodds: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """-log(sigmoid(log_odds))"""
    return np.log1p(np.exp(-lodds))


def CLLR(
    target_llrs: npt.NDArray[np.float_], nontarget_llrs: npt.NDArray[np.float_]
) -> float:
    """
    Compute Log Likelihood Ratio Cost (Cllr) metrics.
    Cllr = 0 indicates perfection while Cllr = 1 indicates an uninformative system.

    Args:
        target_llrs (npt.NDArray[np.float_]): tensor of llr values for target trials,
            i.e. LLR values for entries of the same class.
        nontarget_llrs (npt.NDArray[np.float_]): tensor of llr values for nontarget
            trials, i.e. LLR values for entries of different classes.

    Returns:
        float: Cllr value
    """

    return (
        0.5
        * (
            negative_log_sigmoid(target_llrs).mean()
            + negative_log_sigmoid(-nontarget_llrs).mean()
        )
        / np.log(2)
    )
