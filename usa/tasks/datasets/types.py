from typing import NamedTuple

import torch
from torch import Tensor


class PosedRGBDItem(NamedTuple):
    """Defines a posed RGB image.

    We presume the images and depths to be the distorted images, meaning that
    the depth plane should be flat rather than radial.
    """

    image: Tensor
    depth: Tensor
    mask: Tensor
    intrinsics: Tensor
    pose: Tensor

    def check(self) -> None:
        # Image should have shape (C, H, W)
        assert self.image.dim() == 3
        assert self.image.dtype == torch.float32
        # Depth should have shape (1, H, W)
        assert self.depth.dim() == 3
        assert self.depth.shape[0] == 1
        assert self.depth.dtype == torch.float32
        # Depth shape should match image shape.
        assert self.depth.shape[1:] == self.image.shape[1:]
        assert self.mask.shape[1:] == self.image.shape[1:]
        # Intrinsics should have shape (3, 3)
        assert self.intrinsics.shape == (3, 3)
        assert self.intrinsics.dtype == torch.float64
        # Pose should have shape (4, 4)
        assert self.pose.shape == (4, 4)
        assert self.pose.dtype == torch.float64
