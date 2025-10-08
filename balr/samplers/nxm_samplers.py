import logging
import math
from collections import defaultdict, deque
from collections.abc import Iterator
from itertools import batched, chain

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
from hydra.utils import get_class
from torch.utils.data import Sampler

LOGGER = logging.getLogger(__name__)


def get_labels_to_indices(labels: npt.NDArray) -> dict[int, npt.NDArray[np.int_]]:
    labels_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        labels_to_indices[label].append(idx)

    return {
        label: np.array(indices, dtype=int)
        for label, indices in labels_to_indices.items()
    }


class NxMSampler(Sampler[int]):
    """
    Base class for NxM Sampler. A NxM sampler samples a dataset in batches where each
    batch of size N * M contains N classes and M samples per class.

    Different implementations of the NxMSampler can enforce various extra conditions,
    such as ensuring that each batch contains N distinct classes.

    .. warning::
        Calling the :meth:`set_epoch` method at the beginning of each epoch **before**
        creating the :class:`DataLoader` iterator is necessary to make shuffling work
        properly, otherwise the same ordering will be always used.
    """

    def __init__(
        self,
        labels: npt.NDArray,
        N_classes_per_batch: int,
        M_samples_per_class: int,
        num_batches: int,
        total_size: int,
        batch_size: int | None,
        shuffle: bool,
        drop_last: bool,
        num_replicas: int | None,
        rank: int,
        seed: int,
    ):
        self.labels = labels
        self.N_classes_per_batch = N_classes_per_batch
        self.M_samples_per_class = M_samples_per_class
        self.num_batches = num_batches
        self.total_size = total_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed

        if (
            batch_size is not None
            and self.N_classes_per_batch * self.M_samples_per_class != batch_size
        ):
            raise ValueError(
                f"Invalid batch_size ({batch_size}) for sampler. batch_size must be "
                f"equal to N_classes_per_batch ({self.N_classes_per_batch}) * "
                f"M_samples_per_class ({self.M_samples_per_class})."
            )
        self.batch_size = int(self.N_classes_per_batch * self.M_samples_per_class)

        self._setup_distributed(num_replicas, rank)

        LOGGER.debug(self)

    def _setup_distributed(self, num_replicas: int | None, rank: int):
        """
        Setup the sampler for multi-gpu use, when rank > -1.

        Args:
            num_replicas (int | None): number of compute nodes
            rank (int): current rank
        """
        if rank != -1:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    f"Invalid rank {rank}, rank should be in the interval [0, "
                    f"{num_replicas - 1}]"
                )
            self.distributed = True
            self.num_replicas = num_replicas
            self.rank = rank

            # If the nb of batches is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and self.num_batches % self.num_replicas != 0:
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.num_batches_per_replica = math.ceil(
                    (self.num_batches - self.num_replicas) / self.num_replicas
                )
            else:
                self.num_batches_per_replica = math.ceil(
                    self.num_batches / self.num_replicas
                )
        else:
            self.distributed = False
            self.num_replicas = 1
            self.rank = rank

    def _set_labels_to_indices(
        self, labels: npt.NDArray, N_classes_per_batch: int
    ) -> None:
        """
        Group the labels and keep a list of label -> indices.

        Args:
            labels (npt.NDArray): the dataset labels.
            N_classes_per_batch (int): the number of classes per batch

        Raises:
            ValueError: if the number of unique labels is less than N_classes_per_batch.
        """
        self.labels_to_indices = get_labels_to_indices(labels)
        self.unique_labels = np.array(list(self.labels_to_indices.keys()))

        if N_classes_per_batch > len(self.unique_labels):
            raise ValueError(
                f"Not enough classes ({len(self.unique_labels)}) in dataset to sample "
                f"with N_classes_per_batch={N_classes_per_batch}"
            )

    def batch_iter(
        self,
    ) -> Iterator[list[int]]:
        """
        The main sampler method which returns an iterator of batches, each batch
        being N * M samples.
        The method has to be implemented by the subclasses.

        Yields:
            Iterator[list[int]]: the batch iterator.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[int]:
        """
        Create an iterator using this sampler. This function first calls
        `self.batch_iter()` to create a batched iterator with batches of N * M samples.
        Then if `self.distributed` is False, it yields the indices from the successive
        batches. If `self.distributed` is True, it yields from the batches in steps of
        `self.num_replicas`, to distribute the batches between the nodes.

        Yields:
            Iterator[int]: an Iterator over the indices sampled.
        """
        batches = self.batch_iter()
        if self.distributed:
            if self.num_batches_per_replica * self.num_replicas > self.num_batches:
                batches = chain(batches, self.batch_iter())
            for idx, batch in enumerate(batches):
                if (
                    idx % self.num_replicas == self.rank
                    and idx // self.num_replicas < self.num_batches_per_replica
                ):
                    yield from batch
        else:
            for batch in batches:
                yield from batch

    def __len__(self) -> int:
        if self.distributed:
            return self.num_batches_per_replica * self.batch_size

        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different random
        ordering for each epoch. Otherwise, the next iteration of this sampler will
        yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def _get_random_generator(self) -> np.random.Generator:
        """
        When in distributed mode, we need to use a seeded generator to ensure that all
        replicas yield the same ordering for the same epoch, to avoid randomly choosing
        the same samples for different replicas.
        When not in distributed mode, this is not necessary since only one replica will
        do the random permutations so we use an unseeded generator to avoid having to
        call `set_epoch`.

        Returns:
            np.random.Generator: the generator to use for random sampling.
        """
        if self.distributed:
            return np.random.default_rng(self.seed + self.epoch)

        return np.random.default_rng()

    def __str__(self) -> str:
        """
        A string representation of the sampler.

        Returns:
            str: a str describing the sampler's parameters.
        """
        repr_str = (
            f"{self.__class__.__name__}: dataset_len={len(self.labels)}, num_classes="
            f"{len(self.unique_labels)}, N={self.N_classes_per_batch}, M="
            f"{self.M_samples_per_class}, batch_size={self.batch_size}, num_batches="
            f"{self.num_batches}, total_samples={self.total_size}"
        )
        if self.distributed:
            repr_str += (
                f", rank={self.rank}, num_replicas={self.num_replicas}, "
                f"batches_per_replica={self.num_batches_per_replica}."
            )
        else:
            repr_str += "."
        return repr_str


class RandomNxMSampler(NxMSampler):
    """
    An implementation of NxMSampler that only samples each individual class once, always
    sampling M samples per class.

    Thus, the length of the sampler is equal to the number of distinct classes multiplied
    by M. If the number of distinct classes is not divisible by N, either the extra
    classes are omitted if `drop_last` is True, or additional classes are sampled again
    otherwise.

    Each batch will have samples from N classes that are distinct. The M samples for an
    individual class are sampled randomly from all the samples for that class present in
    the dataset. If there are less samples for a class than M, the samples for that class
    will be repeated.

    Example:
        for a dataset with 10 samples and 3 classes:

        >>> labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        >>> sampler = RandomNxMSampler(labels, N_classes_per_batch=2,
        ... M_samples_per_class=2, shuffle=False, drop_last=True)
        >>> list(sampler)
        ... [3, 4, 5, 7]

        The sampler will return 1 batch of 2*2 samples from the first two classes
        (the last class will be droped).

        If drop_last=False, the sampler will return 2 batches of 2*2 samples, the second
        batch having 2 samples from the last class + 2 samples from the first class
        again.

        >>> sampler = RandomNxMSampler(labels, N_classes_per_batch=2,
        ... M_samples_per_class=2, shuffle=False, drop_last=False)
        >>> list(sampler)
        ... [3, 4, 5, 7, 9, 8, 0, 1]

    """

    def __init__(
        self,
        labels: npt.NDArray | torch.Tensor,
        N_classes_per_batch: int,
        M_samples_per_class: int,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        num_replicas: int | None = None,
        rank: int = -1,
        seed=1234,
    ):
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()
        self._set_labels_to_indices(labels, N_classes_per_batch)

        if drop_last and len(self.unique_labels) % N_classes_per_batch != 0:
            num_batches = math.ceil(
                (len(self.unique_labels) - N_classes_per_batch) / N_classes_per_batch
            )
        else:
            num_batches = math.ceil(len(self.unique_labels) / N_classes_per_batch)

        total_size = num_batches * N_classes_per_batch * M_samples_per_class

        super().__init__(
            labels,
            N_classes_per_batch,
            M_samples_per_class,
            num_batches,
            total_size,
            batch_size,
            shuffle,
            drop_last,
            num_replicas,
            rank,
            seed,
        )

    def batch_iter(self) -> Iterator[list[int]]:
        # deterministically shuffle based on epoch and seed
        rng = self._get_random_generator()

        # 1. first shuffle if required
        labels = self.unique_labels
        if self.shuffle:
            labels = rng.permutation(labels)

        # 2. pad labels if required
        if (mod := len(labels) % self.N_classes_per_batch) != 0:
            if self.drop_last:
                labels = labels[:-mod]
            else:
                pad_size = self.N_classes_per_batch - mod
                labels = np.pad(labels, (0, pad_size), "wrap")

        # 3. create batches containing N_classes_per_batch * M_samples_per_class elements
        for batch in batched(labels, self.N_classes_per_batch):
            batch_indices: list[int] = []
            for label in batch:
                samples_for_class = self.labels_to_indices[label]
                replace = len(samples_for_class) < self.M_samples_per_class
                class_indices = rng.choice(
                    samples_for_class, self.M_samples_per_class, replace=replace
                )
                batch_indices.extend(class_indices)
            yield batch_indices


class ExhaustiveNxMSampler(NxMSampler):
    """
    An implementation of NxMSampler that will iterate over elements of the dataset by
    batches of N * M samples, until either all the dataset has been sampled, or there are
    less than N * M samples remaining.

    As much as possible, the batches will have N distinct classes, but this is not
    guaranteed, for instance if there are only samples from a single class remaining.

    Example:
        for a dataset with 10 samples and 3 classes:
        >>> labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        >>> sampler = ExhaustiveNxMSampler(labels, N_classes_per_batch=2,
        ... M_samples_per_class=2, shuffle=False)
        >>> list(sampler)
        ... [0, 1, 5, 6, 8, 9, 2, 3, 7, 5, 4, 0]
    """

    def __init__(
        self,
        labels: npt.NDArray | torch.Tensor,
        N_classes_per_batch: int,
        M_samples_per_class: int,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        num_replicas: int | None = None,
        rank: int = -1,
        seed=1234,
    ):
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()
        self._set_labels_to_indices(labels, N_classes_per_batch)

        # pad the indices for each label to make sure it's divisible by M
        for label in self.unique_labels:
            label_indices = self.labels_to_indices[label]
            if (mod := len(label_indices) % M_samples_per_class) != 0:
                pad_size = M_samples_per_class - mod
                label_indices = np.pad(label_indices, (0, pad_size), "wrap")
                self.labels_to_indices[label] = label_indices

        NxM = int(N_classes_per_batch * M_samples_per_class)
        # make sure total size of dataset is divisible by batch_size, otherwise
        # drop samples to make it divisible
        total_size = sum([len(indices) for indices in self.labels_to_indices.values()])
        if total_size % NxM != 0:
            num_batches = math.ceil((total_size - NxM) / NxM)
        else:
            num_batches = math.ceil(total_size / NxM)
        total_size = num_batches * NxM

        super().__init__(
            labels,
            N_classes_per_batch,
            M_samples_per_class,
            num_batches,
            total_size,
            batch_size,
            shuffle,
            drop_last,
            num_replicas,
            rank,
            seed,
        )

    def batch_iter(self) -> Iterator[list[int]]:
        # deterministically shuffle based on epoch and seed
        rng = self._get_random_generator()

        # 1. first shuffle classes if required
        labels = self.unique_labels
        if self.shuffle:
            labels = rng.permutation(labels)

        indices_for_label: list[deque[int]] = []
        for label in labels:
            label_indices = self.labels_to_indices[label]
            if self.shuffle:
                label_indices = rng.permutation(label_indices)
            indices_for_label.append(deque(label_indices))

        # 2. create batches containing N_classes_per_batch * M_samples_per_class elements
        sorted_idxs: list[int] = []
        while len(sorted_idxs) < self.total_size:
            for indices in indices_for_label:
                if len(indices):
                    for _ in range(self.M_samples_per_class):
                        sorted_idxs.append(indices.popleft())

        # 3. yield from the sorted indices in batches
        yield from batched(sorted_idxs[: self.total_size], self.batch_size)


def build_sampler(
    sampler_path: str,
    labels: npt.NDArray | torch.Tensor,
    N_classes_per_batch: int,
    M_samples_per_class: int,
    batch_size: int | None = None,
    shuffle: bool = True,
    drop_last: bool = False,
    rank: int = -1,
) -> NxMSampler:
    sampler_class = get_class(sampler_path)

    sampler = sampler_class(
        labels=labels,
        N_classes_per_batch=N_classes_per_batch,
        M_samples_per_class=M_samples_per_class,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        rank=rank,
    )

    return sampler
