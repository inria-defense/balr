from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from balr.data.dataset import AudioDataset
from balr.evaluation.metrics import compute_metrics
from tests.conftest import ROOT_DIR


def test_compute_metrics_raises_errors(voxceleb2_metadata: Path):
    dataset = AudioDataset.from_path(voxceleb2_metadata)
    trials = pd.read_csv(ROOT_DIR / "resources/data/voxceleb2/trials.csv", sep="\t")

    with pytest.raises(
        ValueError, match="Trials dataframe does not have a `scores` column."
    ):
        compute_metrics(dataset, trials)

    enroll_id = trials.at[0, "enrollment"]
    other_id = next(filter(lambda x: x != enroll_id, dataset.data_ids))
    trials.at[0, "enrollment"] = f"{enroll_id},{other_id}"
    rng = np.random.default_rng()
    trials["scores"] = rng.random((trials.shape[0],))

    with pytest.raises(
        ValueError, match="Enrollment or test trial with different labels"
    ):
        compute_metrics(dataset, trials)


def test_compute_metrics_returns_dict(voxceleb2_metadata: Path):
    dataset = AudioDataset.from_path(voxceleb2_metadata)
    trials = pd.read_csv(ROOT_DIR / "resources/data/voxceleb2/trials.csv", sep="\t")
    rng = np.random.default_rng()
    trials["scores"] = rng.random((trials.shape[0],))

    metrics = compute_metrics(dataset, trials)

    assert "EER" in metrics
    assert "CLLR" in metrics
