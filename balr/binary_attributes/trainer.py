import gc
import io
import logging
import os
import resource
import time
import traceback
from itertools import chain
from pathlib import Path
from typing import Literal, cast

import numpy.typing as npt
import torch
from psutil import virtual_memory
from sklearn.calibration import LabelEncoder
from torch import distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from balr.binary_attributes.autoencoder import AutoEncoder
from balr.config.configs import DataConfig
from balr.data.dataset import AnnotatedAudio, AudioDataset, build_dataloader
from balr.losses.base_autoencoder_loss import BaseAutoEncoderLoss
from balr.samplers.nxm_samplers import build_sampler
from balr.utils import select_device, set_seed, setup_save_dir

LOGGER = logging.getLogger(__name__)
RANK = int(os.environ.get("RANK", -1))
LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html


def collate_fn(batch: list[AnnotatedAudio]) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels = []
    for sample in batch:
        if sample.embedding is None:
            raise RuntimeError(
                f"Sample {sample.id} does not have an embedding. Training aborted."
            )
        features.append(sample.embedding)
        labels.append(sample.speaker)

    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def init_weights(m: nn.Module):
    """
    Initialize weights for each layer of the model

    Args:
        m (nn.Module): a layer of the model.
    """
    if type(m) is nn.Linear:
        nn.init.kaiming_normal_(m.weight, mode="fan_in")


class BinaryAttributeEncoderTrainer:

    optimizer: optim.Adam
    loss_funcs: list[BaseAutoEncoderLoss]
    model: AutoEncoder | nn.parallel.DistributedDataParallel
    train_loader: DataLoader
    val_loader: DataLoader | None

    def __init__(
        self,
        train: AudioDataset,
        val: AudioDataset,
        data_config: DataConfig,
        loss_funcs: list[BaseAutoEncoderLoss],
        input_dim: int = 256,
        internal_dim: int = 512,
        learning_rate: float = 0.001,
        epochs: int = 100,
        seed: int = 1234,
        save_dir: Path | None = None,
        exist_ok: bool = False,
        save_period: int = 0,
        log_period: int = 2,
        val_period: int = 10,
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_dir = save_dir
        self.exist_ok = exist_ok
        self.save_period = save_period
        self.log_period = log_period
        self.val_period = val_period
        self.input_dim = input_dim
        self.internal_dim = internal_dim
        self.loss_funcs = loss_funcs
        self.loss_labels = tuple(func.name for func in self.loss_funcs)

        self.trainset = train
        self.valset = val
        self.data_config = data_config
        self.N_classes_per_batch = data_config.N_classes_per_batch
        self.M_samples_per_class = data_config.M_samples_per_class

        if RANK in {-1, 0}:
            self._setup_writers(exist_ok=exist_ok)

        self._setup_ddp(device, data_config.batch_size)
        self.seed = seed
        set_seed(self.seed)

        self.best_fitness = None
        self.fitness = None

    def train(self):
        """
        Run training on the train set for given number of epochs.
        """
        try:
            self._setup_train()

            self.start_epoch = 0
            self.train_time_start = time.time()
            LOGGER.info(
                "<================ Training process ================>\n"
                f"{self}\n"
                "<==================================================>"
            )
            self.optimizer.zero_grad()

            for epoch in range(self.epochs):
                self.epoch = epoch
                self.fitness = None
                self._run_epoch(epoch)

                if RANK in {-1, 0}:
                    final_epoch = epoch >= self.epochs - 1

                    if final_epoch or (
                        self.val_period and not (epoch + 1) % self.val_period
                    ):
                        self.validate()

                    if (self.log_period > 0) and ((epoch + 1) % self.log_period == 0):
                        self._save_metrics(self.train_loss, self.epoch, "train")

                    self.save_model()

                # clear memory if utilization > 50 %
                # unused at the moment because memory consumption is low
                # self._clear_memory(threshold=0.5)

            if RANK in {-1, 0}:
                seconds = time.time() - self.train_time_start
                LOGGER.info(
                    f"{self.epoch - self.start_epoch + 1} epochs completed "
                    f"in {seconds / 3600:.3f} hours.\n"
                    f"Results saved to [bold blue]{self.save_dir}[/].\n"
                    f"Run [bold green]tensorboard --logdir={self.save_dir}[/] "
                    "to show metrics."
                )

            self._cleanup_train()
        except Exception as e:
            LOGGER.error(e)
            LOGGER.error(traceback.format_exc())
            raise e

    def _run_epoch(self, epoch: int):
        """
        Run epoch by processing all data in the train set in batches and
        updating losses.

        Args:
            epoch (int): the epoch
        """
        self.train_loss = torch.zeros(1 + len(self.loss_funcs)).to(self.device)
        self.model.train()
        nb_batches = len(self.train_loader)  # number of batches
        try:
            self.train_loader.sampler.set_epoch(epoch)  # type: ignore
        except AttributeError:
            pass

        pbar = enumerate(self.train_loader)
        if RANK in {-1, 0}:
            nb_of_loss_labels = len(self.loss_labels)
            LOGGER.info(
                ("%s" + "%11s" * (nb_of_loss_labels + 2))
                % ("Epoch", "Memory", "Loss", *self.loss_labels)
            )
            pbar = tqdm(enumerate(self.train_loader), total=nb_batches)
        for i, (features, labels) in pbar:
            self.global_step = i + nb_batches * epoch
            self.optimizer.zero_grad()
            loss_items = self._process_batch(features, labels)
            total_loss = torch.sum(
                torch.stack(
                    [
                        loss_item * func.weight
                        for loss_item, func in zip(loss_items, self.loss_funcs)
                    ]
                )
            )
            self.train_loss += torch.hstack([total_loss] + loss_items)
            total_loss.backward()
            self.optimizer.step()

            if RANK in {-1, 0}:
                nb_of_loss_items = len(loss_items)
                cast(tqdm, pbar).set_description(
                    ("%25s%11s" + "%11.4g" * (nb_of_loss_items + 1))
                    % (
                        f"{epoch + 1}/{self.epochs}",
                        f"{self._get_memory():.3g}G",  # (GB) GPU memory util,
                        total_loss,
                        *loss_items,
                    )
                )

        self.train_loss = self.train_loss.cpu() / len(self.train_loader)

    def _process_batch(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Process a batch of samples using the model. Can be called either during
        training or validation. Returns the computed losses on the processed batch.

        Args:
            features (torch.Tensor): input batch features
            labels (torch.Tensor): input batch labels

        Returns:
            list[torch.Tensor]: the computed losses.
        """
        features = features.to(torch.float).to(self.device)
        labels = labels.to(self.device)
        recon, binary, Z = self.model(features)

        losses: list[torch.Tensor] = [
            func(features, labels, binary, recon, Z) for func in self.loss_funcs
        ]
        losses = [loss.to(self.device) for loss in losses]
        return losses

    def validate(self):
        """
        Run validation on test set.

        Raises:
            RuntimeError: if called when test set has not been loaded (multi-gpu case).
        """
        if self.val_loader is None:
            raise RuntimeError(
                f"Trying to run validation on device with rank {RANK}."
                "Validation dataloader has not been initialized."
            )

        self.model.eval()
        self.eval_loss = torch.zeros(1 + len(self.loss_funcs)).to(self.device)
        LOGGER.info("Validating model...")
        with torch.no_grad():
            for features, labels in self.val_loader:
                loss_items = self._process_batch(features, labels)
                total_loss = torch.sum(
                    torch.stack(
                        [
                            loss_item * func.weight
                            for loss_item, func in zip(loss_items, self.loss_funcs)
                        ]
                    )
                )
                self.eval_loss += torch.hstack([total_loss] + loss_items)

        self.eval_loss = self.eval_loss.cpu() / len(self.val_loader)
        self.fitness = (
            -self.eval_loss[0].detach().cpu().numpy()
        )  # use total loss as fitness measure
        if self.best_fitness is None or self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
        self._save_metrics(self.eval_loss, self.epoch, "test")

        loss_log = " ".join(
            [
                f"Test/{label}: {loss:.4g}."
                for label, loss in zip(self.loss_labels, self.eval_loss[1:])
            ]
        )
        LOGGER.info(f"Validation done. Test/Loss: {self.eval_loss[0]:.4g}. {loss_log}")

    def save_model(self):
        """
        Save model training checkpoints.
        """
        if self.save_dir is None:
            return

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        last = self.save_dir / "last.pt"
        last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness is not None and self.best_fitness == self.fitness:
            best = self.save_dir / "best.pt"
            best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and ((self.epoch + 1) % self.save_period == 0):
            (self.save_dir / f"epoch{self.epoch}.pt").write_bytes(
                serialized_ckpt
            )  # save epoch

    def _save_metrics(
        self, losses: torch.Tensor, epoch: int, mode: Literal["train", "test"]
    ):
        """
        Save loss metrics to disk using SummaryWriter.

        Args:
            losses (torch.Tensor): the loss metrics.
            epoch (int): the current epoch.
            mode (Literal[&quot;train&quot;, &quot;test&quot;]): the mode.
        """
        loss_labels = ("total_loss",) + self.loss_labels
        for label, loss in zip(loss_labels, losses):
            self.writer.add_scalar(mode + "/" + label, loss.item(), epoch)

    def get_dataloader(
        self,
        dataset: AudioDataset,
        data_config: DataConfig,
        rank: int,
        mode: Literal["train", "test"] = "train",
    ) -> DataLoader:
        """
        Prepare data for train or test sets, by encoding the speaker ids as int labels.
        Labels (speaker ids) are only used to group embeddings by speaker, thus we
        can use a different LabelEncoder for train and test sets.

        The sampler class provided in the data_config is responsible for sorting the
        dataset and respecting the constraints required for the loss functions used
        during training.

        Returns a dataloader on the encoded dataset.

        Args:
            dataset (AudioDataset): the dataset to load.
            data_config (DataConfig): the DataConfig for the dataloader.
            rank (int): Process rank for distributed training.
            mode (Literal[&quot;train&quot;, &quot;test&quot;], optional):
                type of dataset. Defaults to "train".

        Returns:
            DataLoader: a dataloader on the encoded dataset.
        """
        LOGGER.info(f"Encoding labels for {mode} dataset...")
        # 1. Create label encoder and fit on speaker ids
        speaker_ids = [item["speaker"] for _, item in dataset.iter_dicts()]
        encoder = LabelEncoder()
        labels: npt.NDArray[int] = encoder.fit_transform(speaker_ids)  # type: ignore
        if mode == "train":
            self.nb_train_classes = len(encoder.classes_)

        # 2. Create dataset with speaker ids as int labels
        data = {}
        idx = 0
        for id, item in dataset.iter_dicts():
            data[id] = {**item, "speaker": labels[idx]}
            idx += 1
        encoded_dataset = AudioDataset(data, dataset.data_ids, sort=False)
        LOGGER.info("Done.")

        # 3. Create sampler from data_config
        sampler = None
        if data_config.sampler is not None:
            sampler = build_sampler(
                data_config.sampler,
                labels,
                data_config.N_classes_per_batch,
                data_config.M_samples_per_class,
                data_config.batch_size,
                data_config.shuffle,
                False,
                rank,
            )

        # 4. Build and return dataloader
        dataloader = build_dataloader(
            encoded_dataset,
            data_config.batch_size,
            data_config.num_workers,
            data_config.shuffle,
            rank,
            self.device,
            collate_fn,
            sampler,
        )
        return dataloader

    def _setup_train(self) -> None:
        """
        Setup training by calling self._setup_model, self._prepare_data,
        self._setup_loss and self._build_optimizer.
        """
        LOGGER.info("<================ Setup Training ================>")
        self._setup_model(self.input_dim, self.internal_dim)

        rank = LOCAL_RANK if self.world_size > 1 else -1
        self.train_loader = self.get_dataloader(
            self.trainset, self.data_config, rank, "train"
        )
        if RANK in {-1, 0}:
            self.val_loader = self.get_dataloader(
                self.valset, self.data_config, -1, "test"
            )
        self._setup_loss()
        self._build_optimizer()
        LOGGER.info("<================ Trainer Ready ================>")

    def _build_optimizer(
        self,
    ) -> None:
        """
        Setup the optimizer with the model params and optional loss function params
        if the loss function setup is ArcFaceLoss.

        Raises:
            RuntimeError: if model has not been initialized yet.
        """
        if not hasattr(self, "model"):
            raise RuntimeError(
                "Model not initialized. You must run `_setup_model` "
                "before running `_build_optimizer`."
            )

        params = chain(
            self.model.parameters(), *[func.parameters() for func in self.loss_funcs]
        )
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

    def _setup_model(self, input_dim: int, internal_dim: int) -> None:
        """
        Setup the model to train (AutoEncoder). Sets initial_weights on model
        by calling `init_weights`.

        Args:
            input_dim (int): the input (embedding) dimension
            internal_dim (int): the latent space dimension
        """
        self.model = AutoEncoder(input_dim, internal_dim)
        self.model.apply(init_weights)
        self.model = self.model.to(self.device)
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[RANK]
            )

    def _setup_loss(self) -> None:
        """
        Setup loss functions.
        """
        for func in self.loss_funcs:
            func.setup(
                input_dim=self.input_dim,
                internal_dim=self.internal_dim,
                nb_train_classes=self.nb_train_classes,
                M_samples_per_class=self.M_samples_per_class,
            )
            func.to(self.device)

    def _setup_writers(self, exist_ok: bool = False):
        """
        Create self.save_dir if it does not exist and increments it if it does.
        Also sets up the SummaryWriter used during training.

        Args:
            exist_ok (bool, optional): If True, the path will not be incremented and
                returned as-is. Defaults to False.
        """
        self.save_dir = setup_save_dir(
            self.save_dir, exist_ok=exist_ok, default_path="runs/train"
        )
        LOGGER.info(f"Saving results to [bold blue]{self.save_dir}[/].")
        self.writer = SummaryWriter(self.save_dir)

    def _setup_ddp(self, device: str | torch.device, batch: int):
        """
        Select the device based on input parameter and initialize and set the
        DistributedDataParallel parameters for training if world size > 1 (multi-gpus
        selected).

        Args:
            device (str | torch.device): requested device
        """
        self.device = select_device(device, batch)

        if isinstance(device, str) and len(device):  # device='0' or device='cuda:1,3'
            self.world_size = len(device.split(","))
        elif torch.cuda.is_available():  # device=None or device='' or device=number
            self.world_size = 1
        else:
            self.world_size = 0

        if self.world_size > 1:
            torch.cuda.set_device(RANK)
            self.device = torch.device("cuda", RANK)
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
            dist.init_process_group(
                backend="nccl" if dist.is_nccl_available() else "gloo",
                rank=RANK,
                world_size=self.world_size,
            )

    def _cleanup_train(
        self,
    ):
        if self.world_size > 1:
            dist.destroy_process_group()

    def _get_memory(self, fraction=False) -> float:
        """
        Get accelerator memory utilization in GB or as a fraction of total memory.

        Args:
            fraction (bool, optional): get memory usage as a fraction.
                Defaults to False.

        Returns:
            float: memory usage.
        """
        memory, total = 0.0, 0.0
        if self.device.type == "cpu":
            memory = virtual_memory().used
            if fraction:
                return virtual_memory().percent / 100
            memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        elif self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return virtual_memory().percent / 100
        else:
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float | None = None):
        """
        Clear accelerator memory by calling garbage collector and emptying cache.

        Args:
            threshold (float | None, optional): only clear memory if current usage is
                above threshold. Defaults to None.
        """
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        LOGGER.debug("Clearing memory.")
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def __str__(self) -> str:
        repr_str = (
            f"{self.__class__.__name__}: epochs={self.epochs}, lr={self.learning_rate}, "
            f"losses={self.loss_labels}, val_period={self.val_period}.\n"
            f"{self.trainset}\n"
        )
        if hasattr(self, "train_loader"):
            repr_str += (
                "DataLoader: workers="
                f"{self.train_loader.num_workers * (self.world_size or 1)}.\n"
            )
            if self.train_loader.sampler is not None:
                repr_str += f"{self.train_loader.sampler}\n"
        repr_str += (
            f"DDP: RANK {RANK}, WORLD_SIZE {self.world_size}, DEVICE {self.device}"
        )
        return repr_str
