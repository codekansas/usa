import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import ml.api as ml
import more_itertools
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data.dataset import Dataset

from usa.tasks.datasets.home_robot import (
    HomeRobotDataset,
    chris_lab_home_robot_dataset,
)
from usa.tasks.datasets.r3d import LabR3DDataset, StudioR3DDataset
from usa.tasks.datasets.replica_cad import ReplicaCADDataset
from usa.tasks.datasets.stretch import (
    chess_stretch_dataset,
    kitchen_stretch_dataset,
    lab_stretch_dataset,
)
from usa.tasks.datasets.types import PosedRGBDItem

logger = logging.getLogger(__name__)


def get_posed_rgbd_dataset(
    key: str,
    *,
    path: str | Path | None = None,
    img_dim: int | None = None,
    random_crop: bool = True,
) -> Dataset[PosedRGBDItem]:
    if key == "home_robot":
        if path is None:
            path = os.environ.get("HOME_ROBOT_DS_PATH")
        assert path is not None, "Path must be specified for `home_robot` dataset; set `HOME_ROBOT_DS_PATH` env var"
        return HomeRobotDataset(path)
    if key == "chris_lab":
        return chris_lab_home_robot_dataset()
    if key == "lab_r3d":
        return LabR3DDataset(img_dim=img_dim, random_crop=random_crop)
    if key == "studio_r3d":
        return StudioR3DDataset(img_dim=img_dim, random_crop=random_crop)
    if key == "lab_stretch":
        assert img_dim is None, "`img_dim` and `random_crop` not supported for `lab_stretch`"
        return lab_stretch_dataset()
    if key == "kitchen_stretch":
        assert img_dim is None, "`img_dim` and `random_crop` not supported for `kitchen_stretch`"
        return kitchen_stretch_dataset()
    if key == "chess_stretch":
        assert img_dim is None, "`img_dim` and `random_crop` not supported for `chess_stretch`"
        return chess_stretch_dataset()
    if key.startswith("replica_"):
        return ReplicaCADDataset(key[len("replica_") :], img_dim=img_dim, random_crop=random_crop)
    raise NotImplementedError(f"Unsupported dataset key: {key}")


@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def xdiff(self) -> float:
        return self.xmax - self.xmin

    @property
    def ydiff(self) -> float:
        return self.ymax - self.ymin

    @property
    def zdiff(self) -> float:
        return self.zmax - self.zmin

    @classmethod
    def from_arr(cls, bounds: np.ndarray | Tensor) -> "Bounds":
        assert bounds.shape == (3, 2), f"Invalid bounds shape: {bounds.shape}"

        return Bounds(
            xmin=bounds[0, 0].item(),
            xmax=bounds[0, 1].item(),
            ymin=bounds[1, 0].item(),
            ymax=bounds[1, 1].item(),
            zmin=bounds[2, 0].item(),
            zmax=bounds[2, 1].item(),
        )

    @classmethod
    def merge_bounds(cls, a: Tensor, b: Tensor) -> Tensor:
        return torch.stack(
            (
                torch.minimum(a[:, 0], b[:, 0]),
                torch.maximum(a[:, 1], b[:, 1]),
            ),
            dim=1,
        )

    @classmethod
    def from_xyz(cls, xyz: Tensor) -> Tensor:
        assert xyz.shape[-1] == 3

        xmin, xmax = torch.aminmax(xyz[..., 0])
        ymin, ymax = torch.aminmax(xyz[..., 1])
        zmin, zmax = torch.aminmax(xyz[..., 2])

        return torch.tensor([[xmin, xmax], [ymin, ymax], [zmin, zmax]], device=xyz.device, dtype=xyz.dtype)


def get_xyz(depth: Tensor, mask: Tensor, pose: Tensor, intrinsics: Tensor) -> Tensor:
    """Returns the XYZ coordinates for a set of points.

    Args:
        depth: The depth array, with shape (B, 1, H, W)
        mask: The mask array, with shape (B, 1, H, W)
        pose: The pose array, with shape (B, 4, 4)
        intrinsics: The intrinsics array, with shape (B, 3, 3)

    Returns:
        The XYZ coordinates of the projected points, with shape (B, H, W, 3)
    """

    (bsz, _, height, width), device, dtype = depth.shape, depth.device, depth.dtype

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=device, dtype=dtype),
        torch.arange(0, height, device=device, dtype=dtype),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1).flatten(0, 1).unsqueeze(0).repeat_interleave(bsz, 0)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Applies intrinsics and extrinsics.
    xyz = xyz @ intrinsics.inverse().transpose(-1, -2)
    xyz = xyz * depth.flatten(1).unsqueeze(-1)
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[..., None, :3, 3]

    # Mask out bad depth points.
    xyz = xyz.unflatten(1, (height, width))
    xyz[mask.squeeze(1)] = 0.0

    return xyz


def iter_xyz(ds: Dataset[PosedRGBDItem], desc: str, chunk_size: int = 16) -> Iterator[Tuple[Tensor, Tensor]]:
    """Iterates XYZ points from the dataset.

    Args:
        ds: The dataset to iterate points from
        desc: TQDM bar description
        chunk_size: Process this many frames from the dataset at a time

    Yields:
        The XYZ coordinates, with shape (B, H, W, 3), and a mask where a value
        of True means that the XYZ coordinates should be ignored at that
        point, with shape (B, H, W)
    """

    device = ml.AutoDevice.detect_device()
    ds_len = len(ds)  # type: ignore

    for inds in more_itertools.chunked(tqdm.trange(ds_len, desc=desc), chunk_size):
        depth, mask, pose, intrinsics = (
            torch.stack(ts, dim=0)
            for ts in zip(
                *((device.tensor_to(t) for t in (i.depth, i.mask, i.pose, i.intrinsics)) for i in (ds[i] for i in inds))
            )
        )
        xyz = get_xyz(depth, mask, pose, intrinsics)
        yield xyz, mask.squeeze(1)


def get_poses(ds: Dataset[PosedRGBDItem], cache_dir: Optional[Path] = None) -> np.ndarray:
    # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
    cache_loc = None if cache_dir is None else cache_dir / "poses.npy"

    if cache_loc is not None and cache_loc.is_file():
        return np.load(cache_loc)

    all_poses: List[np.ndarray] = []
    for item in tqdm.tqdm(ds, desc="Poses"):
        all_poses.append(item.pose.cpu().numpy())

    poses = np.stack(all_poses)
    if cache_loc is not None:
        cache_loc.parent.mkdir(exist_ok=True, parents=True)
        np.save(cache_loc, poses)

    return poses


def get_pose_bounds(ds: Dataset[PosedRGBDItem], cache_dir: Optional[Path] = None) -> Bounds:
    # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
    cache_loc = None if cache_dir is None else cache_dir / "pose_bounds.npy"

    bounds: Optional[np.ndarray] = None

    if cache_loc is not None and cache_loc.is_file():
        bounds = np.load(cache_loc)
    else:
        for item in tqdm.tqdm(ds, desc="Pose bounds"):
            xyz = item.pose[..., :3, 3].cpu().numpy()
            if bounds is None:
                bounds = np.stack((xyz, xyz), axis=1)
            else:
                bounds[:, 0] = np.minimum(bounds[:, 0], xyz)
                bounds[:, 1] = np.maximum(bounds[:, 1], xyz)
        assert bounds is not None, "No samples found"
        if cache_loc is not None:
            cache_loc.parent.mkdir(exist_ok=True, parents=True)
            np.save(cache_loc, bounds)

    assert bounds is not None, "No samples found"
    return Bounds.from_arr(bounds)


def get_bounds(ds: Dataset[PosedRGBDItem], cache_dir: Optional[Path] = None) -> Bounds:
    # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
    cache_loc = None if cache_dir is None else cache_dir / "bounds.npy"

    bounds: Optional[np.ndarray] = None

    if cache_loc is not None and cache_loc.is_file():
        bounds = np.load(cache_loc)
    else:
        for xyz, mask in iter_xyz(ds, "Bounds"):
            xyz_flat = xyz[~mask]
            minv_torch, maxv_torch = xyz_flat.aminmax(dim=0)
            minv, maxv = minv_torch.cpu().numpy(), maxv_torch.cpu().numpy()
            if bounds is None:
                bounds = np.stack((minv, maxv), axis=1)
            else:
                bounds[:, 0] = np.minimum(bounds[:, 0], minv)
                bounds[:, 1] = np.maximum(bounds[:, 1], maxv)
        assert bounds is not None, "No samples found"
        if cache_loc is not None:
            cache_loc.parent.mkdir(exist_ok=True, parents=True)
            np.save(cache_loc, bounds)

    assert bounds is not None, "No samples found"
    return Bounds.from_arr(bounds)
