from pathlib import Path

import numpy as np
import pandas as pd

from balr.data.dataset import AudioDataset
from balr.scoring.utils import get_positive_negative_scores, get_trial_labels
from tests.conftest import ROOT_DIR


def test_get_trials_labels(voxceleb2_metadata: Path):
    dataset = AudioDataset.from_path(voxceleb2_metadata)
    trials = pd.read_csv(ROOT_DIR / "resources/data/voxceleb2/trials.csv", sep="\t")

    labels = get_trial_labels(dataset, trials)

    expected_labels = (trials["enrollment"] == trials["test"]).to_numpy().astype(int)

    assert all(labels == expected_labels)


def test_get_positive_negative_scores():
    dim = 1234
    rng = np.random.default_rng()
    scores = rng.random((dim,))
    labels = rng.integers(2, size=dim)

    positive_scores, negative_scores = get_positive_negative_scores(scores, labels)

    expected_positive = scores[labels.astype(bool)]
    expected_negative = scores[~labels.astype(bool)]

    assert all(positive_scores == expected_positive)
    assert all(negative_scores == expected_negative)
