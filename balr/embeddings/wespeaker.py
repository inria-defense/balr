import logging
import time
from pathlib import Path

import numpy.typing as npt
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from huggingface_hub import snapshot_download
from tqdm import tqdm
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint

from balr.config.configs import DataConfig, FeaturesConfig
from balr.data.dataset import AnnotatedAudio, AudioDataset, VectorType, build_dataloader
from balr.embeddings.embeddings import EmbeddingsModel
from balr.utils import select_device

LOGGER = logging.getLogger(__name__)


def collate_fn(batch: list[AnnotatedAudio]) -> list[AnnotatedAudio]:
    return batch


class WespeakerModel(EmbeddingsModel):
    def __init__(
        self,
        model_repo: str = "Wespeaker/wespeaker-voxceleb-resnet34-LM",
        model_name: str = "avg_model",
        config_name: str = "config.yaml",
        model_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        features: FeaturesConfig = FeaturesConfig(),
    ):
        """
        Initialize the WespeakerModel class. Downloads the model weights from
        HuggingFace Hub using `model_repo` if necessary.
        Optionally stores the downloaded model into local folder `model_dir`.

        Args:
            model_repo (str, optional): model repository on HuggingFace Hub.
                Defaults to "Wespeaker/wespeaker-voxceleb-resnet34-LM".
            model_name (str, optional): model weights file name within the repository.
                Defaults to "avg_model".
            config_name (str, optional): config file name within the repository.
                Defaults to "config.yaml"
            model_dir (str | Path | None, optional): local folder path to save the
                model files in. Defaults to None.
            device (str, optional): device to run predictions on. Defaults to "cpu".
        """
        # 1. Download model from HF if necessary
        self.model_repo = model_repo

        try:
            LOGGER.info(f"Downloading model {model_repo}.")
            model_dir = Path(snapshot_download(model_repo, local_dir=model_dir))
        except Exception as e:
            LOGGER.info(
                (
                    f"Failed to download wespeaker model or find the model "
                    f"in the cache {model_repo}: {e}."
                )
            )

            if model_dir is not None:
                check_inputs = [Path(model_dir), Path(model_dir) / model_name]

                if not (check_inputs[0].is_dir() and check_inputs[1].is_file()):
                    raise AttributeError(f"{model_dir} or {model_name} don't exists")
            else:
                LOGGER.info("model_dir is missing: set a model folder!")
                raise AttributeError(
                    (
                        "Set a model folder and a model name: "
                        "embeddings.model.model_dir and "
                        "embeddings.model.model_name are missing! "
                    )
                )

        model_dir = Path(model_dir)

        LOGGER.info(f"Model saved in {model_dir.absolute()}.")

        # 2. save device
        self.device = select_device(device)
        self.features = features

        # 3. Load wespeaker model using config and model weights
        config_path = model_dir / config_name
        with open(config_path, "r") as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        self.model = get_speaker_model(configs["model"])(**configs["model_args"])
        load_checkpoint(self.model, model_dir / model_name)
        self.model.to(self.device).eval()

    def extract_embeddings(
        self, dataset: AudioDataset, stream_save: bool, data_config: DataConfig
    ) -> list[tuple[str, npt.NDArray]]:
        """
        Extract embeddings following https://github.com/wenet-e2e/wespeaker/blob/
        310a15850895b54e20845e107b54c9a275d39a2d/wespeaker/bin/extract.py#L33

        Args:
            dataset (AudioDataset): the dataset to extract embeddings for.
            stream_save (bool): if True, save the embeddings to disk as they are
                computed, returning an empty list. Otherwise, return a list of
                embedding and id tuples.
            data_config (DataConfig): dataloader config.

        Returns:
            list[tuple[str, npt.NDArray]]: list of id and embedding tuples if
                stream_save is False, otherwise an empty list (results are saved to the
                dataset during processing).
        """
        dataloader = build_dataloader(
            dataset,
            data_config.batch_size,
            data_config.num_workers,
            data_config.shuffle,
            -1,
            self.device,
            collate_fn,
        )

        LOGGER.info(
            "<================ Embedding Inference ================>\n"
            f"{self}\n{dataset}\n"
            f"DataLoader: batch_size={dataloader.batch_size}, num_batches="
            f"{len(dataloader)}, workers={dataloader.num_workers}\n"
            "<=====================================================>"
        )
        start_time = time.time()
        embeddings: list[tuple[str, npt.NDArray]] = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                for sample in batch:
                    if sample.audio is None:
                        LOGGER.warning(
                            f"Sample {sample.id} does not have an audio signal. "
                            "Embedding inference skipped."
                        )
                        continue
                    waveform = sample.audio.to(torch.float).to(self.device)
                    # 1. compute fbank features
                    features = self.compute_features(
                        waveform,
                        dataset.sample_rate,
                        norm_mean=data_config.normalize_mean,
                        norm_var=data_config.normalize_var,
                        num_mel_bins=self.features.num_mel_bins,
                        frame_length=self.features.frame_length,
                        frame_shift=self.features.frame_shift,
                        dither=self.features.dither,
                    )
                    features = features.unsqueeze(0).to(torch.float).to(self.device)

                    # 2. forward pass through model, extracting embedding.
                    outputs = self.model(features)
                    embedding = outputs[-1] if isinstance(outputs, tuple) else outputs
                    embedding = embedding[0].cpu().detach().numpy()

                    # 3. save embedding to disk or add to results list
                    if stream_save:
                        dataset.save_vector(
                            sample.id, embedding, vector_type=VectorType.embedding
                        )
                    else:
                        embeddings.append((sample.id, embedding))

        seconds = time.time() - start_time
        LOGGER.info(f"Embedding inference completed in {seconds / 3600:.3f} hours.")
        return embeddings

    def compute_features(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        norm_mean: bool = True,
        norm_var: bool = False,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute features for given waveform using kaldi.fbank.

        Args:
            waveform (torch.Tensor): tensor of raw audio signal.
            sample_rate (int): sample rate for audio signal.
            norm_mean (bool, optional): apply cepstral mean normalization.
                Defaults to True.
            norm_var (bool, optional): apply cepstral variance normalization.
                Defaults to False.
            num_mel_bins (int, optional): Number of triangular mel-frequency bins.
                Defaults to 80.
            frame_length (int, optional): Frame length in milliseconds. Defaults to 25.
            frame_shift (int, optional): Frame shift in milliseconds. Defaults to 10.
            dither (float, optional): Dithering constant (0.0 means no dither).
                Defaults to 1.0.

        Returns:
            torch.Tensor: tensor of computed fbank features
        """
        feature = kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            sample_frequency=sample_rate,
            window_type="hamming",
            use_energy=False,
        )
        if norm_mean:
            feature = feature - torch.mean(feature, dim=0)
        if norm_var:
            feature = feature / torch.sqrt(torch.var(feature, dim=0) + 1e-8)
        return feature

    def __str__(self):
        repr_str = (
            f"{self.__class__.__name__}: model={self.model_repo}, device="
            f"{self.device}."
        )
        return repr_str
