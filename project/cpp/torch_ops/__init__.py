# type: ignore
"""Defines the C++ backend for the attention library."""

from pathlib import Path

import torch
from torch import Tensor

from .torch_ops import *  # noqa: F401,F403

# Registers the shared library with Torch.
torch.ops.load_library(Path(__file__).parent.resolve() / "torch_ops.so")

# Access modules registered with `TORCH_LIBRARY` instead of `PYBIND11_MODULE`
TORCH = torch.ops.torch_ops


def nucleus_sampling(logits: Tensor, nucleus_prob: float) -> Tensor:
    # Short-hand wrapper around TorchScript nucleus sampling kernel to
    # provide typing support.
    return TORCH.nucleus_sampling(logits, nucleus_prob)


# Cleans up variables which shouldn't be exported.
del torch, Path
