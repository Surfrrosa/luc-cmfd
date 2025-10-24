"""Utilities for deterministic execution, timing, and logging."""

import random
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_all_seeds(seed: int, logger: logging.Logger = None) -> None:
    """
    Set seeds for Python, NumPy, PyTorch, and CUDNN for reproducibility.

    Args:
        seed: Random seed value
        logger: Optional logger to record seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Limit CPU threads for determinism
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if logger:
        logger.info(f"Set all seeds to {seed}")
        logger.info(f"CPU threads limited to 1 for determinism")


@contextmanager
def Timer(name: str = "Operation", logger: logging.Logger = None):
    """
    Context manager for timing operations.

    Args:
        name: Name of the operation being timed
        logger: Optional logger instance. If None, prints to stdout.

    Yields:
        dict with 'elapsed' key that will be populated after completion
    """
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed"] = elapsed
        msg = f"{name}: {elapsed:.4f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg)


def setup_logger(name: str = "luc-cmfd", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def print_env_banner(logger: logging.Logger = None) -> None:
    """Print environment information banner."""
    import cv2
    import skimage

    # Get versions
    torch_version = torch.__version__
    cv2_version = cv2.__version__
    skimage_version = skimage.__version__

    # Get GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        amp_enabled = torch.cuda.is_bf16_supported() or torch.cuda.get_device_capability()[0] >= 7
    else:
        gpu_name = "CPU only"
        gpu_count = 0
        amp_enabled = False

    banner = (
        f"Environment: torch={torch_version} cv2={cv2_version} skimage={skimage_version} | "
        f"GPU={gpu_name} (count={gpu_count}) AMP={'ON' if amp_enabled else 'OFF'}"
    )

    if logger:
        logger.info(banner)
    else:
        print(banner)


def memory_stats(device: torch.device = None) -> Dict[str, float]:
    """
    Get current GPU memory statistics.

    Args:
        device: CUDA device (default: current device)

    Returns:
        Dict with allocated and reserved memory in GB
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0}

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved
    }
