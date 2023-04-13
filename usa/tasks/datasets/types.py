from typing import NamedTuple

import torch
from torch import Tensor


class NeRFItem(NamedTuple):
    image: Tensor
    pose: Tensor
    intrinsics: Tensor

    # These are only used during the testing phase, as ground truth for
    # evaluating NeRF reconstruction quality.
    depth: Tensor | None
    ray_dir_high: Tensor | None
    view_dir_high: Tensor | None


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


class CLIPItem(NamedTuple):
    image: PosedRGBDItem
    clip: Tensor

    def check(self) -> None:
        self.image.check()
        # CLIP should have shape (C)
        assert self.clip.dim() == 1


class SDFItem(NamedTuple):
    points: Tensor
    sdf: Tensor

    def check(self) -> None:
        # Points should have shape (N, 3)
        assert self.points.dim() == 2
        assert self.points.shape[1] == 3
        # SDF should have shape (N, 1)
        assert self.sdf.dim() == 2
        assert self.sdf.shape[1] == 1
