from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class FeaturesConfig:
    frontend: str = "fbank"
    num_mel_bins: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    dither: float = 0.0


@dataclass
class WespeakerEmbeddingsConfig:
    _target_: str = "balr.embeddings.wespeaker.WespeakerModel"
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model_repo: str = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
    model_name: str = "avg_model"
    config_name: str = "config.yaml"


@dataclass
class EmbeddingsConfig:
    model: Any = MISSING


@dataclass
class BinaryAutoEncoderConfig:
    _target_: str = "balr.binary_attributes.autoencoder.BinaryAttributeAutoEncoder"
    checkpoint_path: str = "resources/models/BAE/BAE_mse.pt"
    input_dim: int = 256
    internal_dim: int = 512


@dataclass
class BAEncoderConfig:
    model: Any = MISSING


@dataclass
class DataConfig:
    batch_size: int = 64
    num_workers: int = 1
    shuffle: bool = True
    normalize_mean: bool = True
    normalize_var: bool = False
    sampler: str | None = "balr.samplers.nxm_samplers.RandomNxMSampler"
    N_classes_per_batch: int = 8
    M_samples_per_class: int = 8


@dataclass
class SparsityLossConfig:
    _target_: str = "balr.losses.sparsity_loss.SparsityLoss"
    weight: float = 0.0001


@dataclass
class MSELossConfig:
    _target_: str = "balr.losses.mse_loss.MSELoss"
    weight: float = 1.0


@dataclass
class TripletMarginLossConfig:
    _target_: str = "balr.losses.triplet_margin_loss.TripletMarginLoss"
    margin: float = 0.3
    type_of_triplets: str = "all"
    weight: float = 1.0


@dataclass
class ArcFaceLossConfig:
    _target_: str = "balr.losses.arcface_loss.ArcFaceLoss"
    margin: float = 0.3
    type_of_triplets: str = "all"
    weight: float = 0.001


@dataclass
class LossConfig:
    """
    Convenience class to select the list of loss functions used during training.
    Hydra does not manage selecting lists easily.
    See https://hydra.cc/docs/patterns/select_multiple_configs_from_config_group/
    and https://github.com/facebookresearch/hydra/issues/1389
    and https://stackoverflow.com/a/79021005

    To override the loss functions, use 'trainer.losses=[mse, triplet]'.
    """

    mse: MSELossConfig | None = None
    triplet: TripletMarginLossConfig | None = None
    sparsity: SparsityLossConfig | None = None
    arcface: ArcFaceLossConfig | None = None


@dataclass
class BATrainerConfig:
    _target_: str = "balr.binary_attributes.trainer.BinaryAttributeEncoderTrainer"
    input_dim: int = 256
    internal_dim: int = 512
    learning_rate: float = 0.001
    epochs: int = 64
    seed: int = 1234
    save_period: int = 0
    log_period: int = 2
    val_period: int = 10
    losses: LossConfig = field(default_factory=LossConfig)


@dataclass
class ScorerConfig:
    checkpoint_path: str | None = None
    n_iterations: int = 100
    n_attributes: int = 512


@dataclass
class CosineScorerConfig(ScorerConfig):
    _target_: str = "balr.scoring.cosine.CosineSimilarityScorer"
    dim: int = 1
    eps: float = 1e-8


@dataclass
class DNAScorerConfig(ScorerConfig):
    _target_: str = "balr.scoring.llr.DNALLRScorer"
    drop_in: float = 0.12
    typicality_threshold: float = 1e-4


@dataclass
class SpeechScorerConfig(ScorerConfig):
    _target_: str = "balr.scoring.llr.SpeechLLRScorer"
    drop_in: float = 0.12
    typicality_threshold: float = 1e-4


@dataclass
class MaxLLRScorerConfig(ScorerConfig):
    _target_: str = "balr.scoring.llr.MaxLLRScorer"
    f: float = 0.5
    p: float = 0.9
    q: float = 0.1


@dataclass
class BetaScorerConfig(ScorerConfig):
    _target_: str = "balr.scoring.beta.DirichletMultinomialScorer"
    K: int = 2
    eps: float = 1e-8


config_defaults = [
    "_self_",
    {"embeddings.model": "wespeaker"},
    {"encoder.model": "autoencoder"},
    {"trainer.losses": ["mse", "triplet"]},
    {"scorer": "beta"},
]
default_audio_formats = ["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"]


@dataclass
class RunConfig:
    input: Path
    save_output: bool = True
    model_dir: Path | None = None
    save_dir: Path | None = None
    exist_ok: bool = False
    audio_formats: list[str] = field(default_factory=lambda: default_audio_formats)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    encoder: BAEncoderConfig = field(default_factory=BAEncoderConfig)
    trainer: BATrainerConfig = field(default_factory=BATrainerConfig)
    scorer: ScorerConfig = MISSING
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "cpu"

    defaults: list[Any] = field(default_factory=lambda: config_defaults)


cs = ConfigStore.instance()
cs.store(name="config", node=RunConfig)
cs.store(group="embeddings.model", name="wespeaker", node=WespeakerEmbeddingsConfig)
cs.store(group="encoder.model", name="autoencoder", node=BinaryAutoEncoderConfig)
cs.store(group="trainer", name="batrainer", node=BATrainerConfig)
cs.store(
    group="trainer.losses",
    name="sparsity",
    node=SparsityLossConfig,
    package="trainer.losses.sparsity",
)
cs.store(
    group="trainer.losses",
    name="mse",
    node=MSELossConfig,
    package="trainer.losses.mse",
)
cs.store(
    group="trainer.losses",
    name="triplet",
    node=TripletMarginLossConfig,
    package="trainer.losses.triplet",
)
cs.store(
    group="trainer.losses",
    name="arcface",
    node=ArcFaceLossConfig,
    package="trainer.losses.arcface",
)
cs.store(group="scorer", name="dna", node=DNAScorerConfig)
cs.store(group="scorer", name="speech", node=SpeechScorerConfig)
cs.store(group="scorer", name="maxllr", node=MaxLLRScorerConfig)
cs.store(group="scorer", name="cosine", node=CosineScorerConfig)
cs.store(group="scorer", name="beta", node=BetaScorerConfig)


def load_run_config(
    input: Path,
    overrides: list[str] | None,
    save_output: bool,
    save_dir: Path | None,
    exist_ok: bool,
    audio_formats: list[str],
    device: str,
    config_path: str | None,
    config_name: str,
) -> RunConfig:
    if overrides is None:
        overrides = []
    overrides += [
        f"input={input}",
        f"audio_formats={audio_formats}",
        f"device='{device}'",
        f"save_output={save_output}",
        f"exist_ok={exist_ok}",
    ]
    if save_dir is not None:
        overrides.append(f"save_dir={save_dir}")

    with initialize(
        version_base=None, config_path=config_path, job_name="embeddings_cli"
    ):
        conf = compose(config_name=config_name, overrides=overrides)
    return cast(RunConfig, conf)
