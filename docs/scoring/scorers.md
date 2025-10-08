# Scorers

The BA-LR toolkit provides different scorer classes which implement the scoring models described in the [previous section](./scoring.md).

## SpeechLLRScorer

```python
class SpeechLLRScorer(LLRScorer):
    def __init__(
        self,
        n_attributes: int = 512,
        drop_in: float = 0.12,
        typicality_threshold: float = 1e-4,
        checkpoint_path: str | Path | None = None,
        save_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ) -> None:
```

**Parameters**:

* **n_attributes**: the number of binary attributes.
* **drop_in**: the drop-in parameter.
* **typicality_threshold**: typicality threshold below which LLR will not be predicted (returns 1).
* **checkpoint_path**: path to the saved model weights.
* **save_dir**: directory where the model weights are saved.
* **device**: the device to use the model on.

## DNALLRScorer

```python
class DNALLRScorer(LLRScorer):
    def __init__(
        self,
        n_attributes: int = 512,
        drop_in: float = 0.12,
        typicality_threshold: float = 1e-4,
        checkpoint_path: str | Path | None = None,
        save_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ) -> None:
```

**Parameters**:

* **n_attributes**: the number of binary attributes.
* **drop_in**: the drop-in parameter.
* **typicality_threshold**: typicality threshold below which LLR will not be predicted (returns 1).
* **checkpoint_path**: path to the saved model weights.
* **save_dir**: directory where the model weights are saved.
* **device**: the device to use the model on.

## MaxLLRScorer

```python
class MaxLLRScorer(Scorer):
    def __init__(
        self,
        n_attributes: int = 512,
        f: float = 0.5,
        p: float = 0.9,
        q: float = 0.1,
        checkpoint_path: str | Path | None = None,
        save_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
```

**Parameters**:

* **n_attributes**: the number of binary attributes.
* **f**: initial value for EM approximation of $f$.
* **p**: initial value for EM approximation of $p$.
* **q**: initial value for EM approximation of $q$.
* **checkpoint_path**: path to the saved model weights.
* **save_dir**: directory where the model weights are saved.
* **device**: the device to use the model on.
* **n_iterations**: the number of iteration steps during EM approximation of the parameters.

## DirichletMultinomialScorer

```python
class DirichletMultinomialScorer(Scorer):
    def __init__(
        self,
        K: int = 2,
        eps: float = 1e-8,
        n_attributes: int = 512,
        checkpoint_path: str | Path | None = None,
        save_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ) -> None:
```

**Parameters**:

* **K**: number of quantized states per attribute (2 for binary).
* **eps**: non zero small value for numerical stability.
* **n_attributes**: the number of binary attributes.
* **checkpoint_path**: path to the saved model weights.
* **save_dir**: directory where the model weights are saved.
* **device**: the device to use the model on.
* **n_iterations**: the number of iteration steps during EM approximation of the parameters.

!!! note

    The Beta-Bernoulli scoring is a special case of a Dirichlet-Multinomial model (with K=2 parameters).

## Usage

The BA-LR cli provides a `score` command to train a scorer on a reference dataset and to score trial pairs.

### Training

The `balr score train` command is used to train a scorer on a given dataset.

**Parameters**:

* **train**: the path to the training dataset.
* **save_dir**: directory where the scorer weights are saved. By default, results will be saved to `./runs/scorer/trainX`, X being incremented as needed (`train2`, `train3`, etc. on successive runs).
* **device**: the device to train the model on.
* **overrides**: optional hydra config overrides, **including the `scorer` parameter which lets you choose which scoring model to train.**

!!! warning

    Since scoring works on binary attribute vectors, make sure that all the samples in the dataset have binary attribute vectors.


!!! example

    ```bash
    balr score train resources/data/voxceleb2/train.csv
    ```

    will train the default scorer (`Beta-Bernoulli`) on the `voxceleb2/train.csv` dataset. The scorer will estimate its parameters over the reference population. The scorer weights will be saved by default to the `./runs/scorer/train` directory. Training will run by default on the `cpu` device.


!!! example

    ```bash
    balr score train resources/data/voxceleb2/train.csv scorer=maxllr scorer.f=0.1 scorer.p=0.7 scorer.q=0.2 scorer.n_iterations=50
    ```

    This more complex command will train a `MaxLLRScorer` on the `voxceleb2/train.csv` dataset, but also modifies the initial values for the scorer's parameters (f, p, q) as well as the number of iterations for the EM approximation algorithm.

### Scoring

The `balr score test` command is used to score trial pairs of recordings using a scoring model.

**Parameters**:

* **test**: the path to the testing dataset.
* **trials**: the path to the trials list.
* **sep**: separator char for trials list. Defaults to "\\t".
* **save_dir**: directory where the scores will be saved. By default, results will be saved to `./runs/scoreX`, X being incremented as needed (`score2`, `score3`, etc. on successive runs).
* **device**: the device to run the model on.
* **overrides**: optional hydra config overrides, **including the checkpoint path of the scorer weights fit on a reference dataset**.

!!! warning

    Most scoring models require to be fitted on a reference population to estimate their distribution parameters (see the training section above). When running scoring, make sure to specify the checkpoint path of the fitted model with the `scorer.checkpoint_path=...` parameter.


The `balr score test` command requires two arguments:

1. a test dataset containing binary attributes vectors for all its samples,
2. a csv list of comma separated ids for the enrollment and test pairs to score.

For example, with a dataset metadata file such as

```csv
id,binary_attributes,speaker
id07417/00028,id07417/00028_ba.txt,id07417
id03184/00022,id03184/00022_ba.txt,id03184
id03184/00053,id03184/00053_ba.txt,id03184
id04961/00169,id04961/00169_ba.txt,id04961
id04961/00289,id04961/00289_ba.txt,id04961
id01184/00133,id01184/00133_ba.txt,id01184
id06261/00159,id06261/00159_ba.txt,id06261
id06261/00233,id06261/00233_ba.txt,id06261
id06261/00190,id06261/00190_ba.txt,id06261
id07531/00142,id07531/00142_ba.txt,id07531
```

which can be loaded with `AudioDataset.from_path`. A trials csv file such as

```csv
enrollment	test
id03184/00022,id03184/00053	id07417/00028
id06261/00159,id06261/00233,id06261/00190	id04961/00169,id04961/00289
```

will score two trials

1. the first comparing the two recordings of speaker `id03184` againt a single recording of speaker `id07417`
2. the second comparing the three recordings of speaker `id06261` againt the two recordings of speaker `id04961`

The `balr score test` command will save the results in a `scores.csv` file in the `runs/scoreX` directory:

```csv
enrollment	test	scores
id03184/00022,id03184/00053	id07417/00028	0.9382
id06261/00159,id06261/00233,id06261/00190	id04961/00169,id04961/00289	0.9817
```

!!! example

    ```bash
    balr score test resources/data/voxceleb2/metadata.csv resources/data/voxceleb2/trials.csv scorer=cosine
    ```

    will score the trial lists in the `resources/data/voxceleb2/trials.csv` file with a `CosineSimilarity` scorer (this scorer does not need to be fitted, thus it does not require a checkpoint_path). The scores will be saved by default to the `./runs/score` directory. Scoring will run by default on the `cpu` device.


!!! example

    ```bash
    balr score test resources/data/voxceleb2/metadata.csv resources/data/voxceleb2/trials.csv scorer=beta scorer.checkpoint_path=runs/scorer/train/scorer.pt
    ```

    This command will score the trial lists in the `resources/data/voxceleb2/trials.csv` file with a `BetaBernouilli` scorer using the scorer weights saved from its training in the `runs/scorer/train/scorer.pt` file.
