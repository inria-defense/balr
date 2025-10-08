import os
import random
from pathlib import Path

import numpy as np
import torch

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # type: ignore


def set_seed(seed=1234):
    """
    Set seed for random functions in various packages.

    Args:
        seed (int, optional): the seed. Defaults to 42.
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def select_device(device: str | torch.device | None, batch: int = 0) -> torch.device:
    """
    Select the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object
    and returns a torch.device object representing the selected device.
    The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device | None): the requested device, such as 'None',
            'cpu', 'cuda', '0', 'cuda:1,3', ... If no device specified (None), defaults
            to first available GPU, or CPU otherwise.
        batch (int, optional): batch size used for the model. Defaults to 0.

    Raises:
        ValueError: if the specified device is not available, or if the batch size is not
            a multiple of the number of devices when using multiple GPUs.

    Returns:
        torch.device: the selected device.
    """
    if isinstance(device, torch.device):
        return device

    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(
            remove, ""
        )  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            "-1"  # force torch.cuda.is_available() = False
        )
    elif device:  # non-cpu device requested
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join(
                [x for x in device.split(",") if x]
            )  # remove sequential commas, i.e. "0,,1" -> "0,1"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            device  # set environment variable - must be before assert is_available()
        )
        if not (
            torch.cuda.is_available()
            and torch.cuda.device_count() >= len(device.split(","))
        ):
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
            )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # i.e. "0,1" -> ["0", "1"]
        n = len(devices)  # device count
        if n > 1:  # multi-GPU
            if batch < 1:
                raise ValueError(
                    "Multi-GPU training with batch size < 1 is not supported. Please"
                    "specify a valid batch size (i.e. batch=16)."
                )
            if (
                batch >= 0 and batch % n != 0
            ):  # check batch_size is divisible by device_count
                raise ValueError(
                    f"'batch={batch}' must be a multiple of GPU count {n}. "
                    f"Try 'batch={batch // n * n}' or 'batch={batch // n * n + n}', "
                    f"the nearest batch sizes evenly divisible by {n}."
                )
        arg = "cuda:0"
    elif mps and torch.backends.mps.is_available():
        # Prefer MPS if available
        arg = "mps"
    else:  # revert to CPU
        arg = "cpu"

    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)  # reset OMP_NUM_THREADS for cpu training
    return torch.device(arg)


def setup_save_dir(
    save_dir: Path | None, exist_ok: bool = False, default_path: str = "runs/train"
) -> Path:
    """
    Create save_dir if it does not exist and increments it if it does.

    Args:
        save_dir (Path | None): directory to create.
        exist_ok (bool, optional): If True, the path will not be incremented and
            returned as-is. Defaults to False.
        default_path (str, optional): default path to use if save_dir is None.
            Defaults to "runs/train".

    Returns:
        Path: the created directory.
    """
    if save_dir is None:
        save_dir = Path(default_path)

    if save_dir.exists() and not exist_ok:
        path, suffix = (
            (save_dir.with_suffix(""), save_dir.suffix)
            if save_dir.is_file()
            else (save_dir, "")
        )

        p = f"{path}{suffix}"
        for n in range(2, 9999):
            p = f"{path}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        save_dir = Path(p)

    save_dir.mkdir(parents=True, exist_ok=True)  # make directory

    return save_dir
