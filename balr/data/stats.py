import logging

import torch
from tqdm import tqdm

from balr.config.configs import DataConfig
from balr.data.dataset import AnnotatedAudio, AudioDataset, build_dataloader

LOGGER = logging.getLogger(__name__)


def collate_fn(batch: list[AnnotatedAudio]) -> list[AnnotatedAudio]:
    return batch


def get_speaker_statistics(
    dataset: AudioDataset, data_config: DataConfig
) -> torch.Tensor:
    """
    Extract a tensor of speaker activation statistics of shape N x n x K,
    with N being the number of speakers , n being the number of attributes, and K being
    the number of values per attribute (2 in the case of binary attributes).

    Args:
        dataset (AudioDataset): the dataset to extract statistics for.
        data_config (DataConfig): dataloader config.

    Returns:
        torch.Tensor: the tensor of activation statistics per speaker.
    """

    dataloader = build_dataloader(
        dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=data_config.shuffle,
        rank=-1,
        device=torch.device("cpu"),
        collate_fn=collate_fn,
    )

    LOGGER.info(
        "Extracting speaker activation statistics from dataset.\n"
        f"{dataset}\nDataLoader: batch_size={dataloader.batch_size}, num_batches="
        f"{len(dataloader)}, workers={dataloader.num_workers}\n"
    )

    speaker_stats: dict[str, torch.Tensor] = dict()

    for batch in tqdm(dataloader, desc="Computing speaker statistics"):
        for sample in batch:
            # check if binary_attributes and speaker exist in the item
            if sample.binary_attributes is None:
                LOGGER.error(
                    f"Item {sample.id} in the dataset does not have a binary "
                    "attributes vector."
                )
                raise ValueError(
                    "An item in the dataset does not have a binary attributes vector. "
                    "Make sure all items in the dataset have binary attributes vectors."
                )

            if sample.speaker is None:
                LOGGER.error(
                    f"Item {sample.id} in the dataset does not have a speaker id."
                )
                raise ValueError(
                    "An item in the dataset does not have a speaker id. "
                    "All items in the dataset need to have speaker ids."
                )

            speaker = sample.speaker
            ba = sample.binary_attributes

            try:
                speaker_stats[speaker] += ba
            except KeyError:
                speaker_stats[speaker] = ba

    activation_stats = torch.stack(list(speaker_stats.values()), dim=0)
    return activation_stats
