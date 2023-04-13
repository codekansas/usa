"""Dataset of some Locobot trajectories in ReplicaCAD.

This is the dataset that was used in the iSDF paper.
"""

import json
import logging
from pathlib import Path

import ml.api as ml
import numpy as np
import torch
import torchvision.transforms.functional as V
from PIL import Image
from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from usa.tasks.datasets.types import PosedRGBDItem
from usa.tasks.datasets.utils import visualize_posed_rgbd_dataset

logger = logging.getLogger(__name__)

# From the Google Drive folder here: https://drive.google.com/drive/folders/1nzAVDInjDwt_GFehyhkOZvXrRJ33FCaR
# EVAL_PTS_URL = "https://drive.google.com/file/d/1ECNbnf9FBCxfWr_IZK_WRwybLKfYyuyE/view?usp=sharing"
# GT_SDFS_URL = "https://drive.google.com/file/d/1qFI2Pd9td8CaRKpSarvOlqdB3b-FRoKw/view?usp=sharing"
# SEQS_URL = "https://drive.google.com/file/d/1IUCymFSKFOno9CRGo6gNnZieb1jDpzoi/view?usp=sharing"

# From the project repo (original ZIP was corrupted for some reason).
SEQS_URL = "https://github.com/codekansas/usa/releases/download/v1.0.0/replica.zip"


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> Tensor:
    intr = np.eye(3, dtype=np.float64)
    intr[0, 0] = fx
    intr[1, 1] = fy
    intr[0, 2] = cx
    intr[1, 2] = cy
    return torch.from_numpy(intr)


def ensure_downloaded(url: str, name: str) -> Path:
    extract_path = ml.get_data_dir() / "ReplicaCAD" / name
    extract_path.parent.mkdir(exist_ok=True, parents=True)
    if extract_path.is_dir():
        logger.debug("Already downloaded %s to %s", name, extract_path)
    else:
        download_and_extract_archive(
            url,
            extract_path.parent,
            extract_root=extract_path.parent,
            filename=f"{name}.zip",
            remove_finished=True,
        )
        assert extract_path.is_dir(), f"Downloading {name} failed"
    return extract_path


class ReplicaCADDataset(Dataset[PosedRGBDItem]):
    def __init__(self, key: str, *, img_dim: int | None = None, random_crop: bool = True) -> None:
        """Defines the ReplicaCAD dataset.

        Args:
            key: Which ReplicaCAD dataset to load
            img_dim: If set, crop image to a square with this image shape
            random_crop: If `img_dim` is set and the image has to be cropped,
                use this option to toggle random verses center cropping
        """

        self.key = key

        # eval_pts_dir = ensure_downloaded(EVAL_PTS_URL, "eval_pts")
        # gt_sdfs_dir = ensure_downloaded(GT_SDFS_URL, "gt_sdfs")
        seqs_dir = ensure_downloaded(SEQS_URL, "seqs")
        self.seq_dir = seqs_dir / key
        assert self.seq_dir.exists(), f"Key {key} not found in {seqs_dir}"

        # Loads the trajectories.
        self.traj = np.loadtxt(self.seq_dir / "traj.txt").reshape(-1, 4, 4)

        # Converts trajectories from (X, Z, Y) to (X, -Y, Z).
        affine_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        self.traj = affine_matrix @ self.traj

        self.num_frames = self.traj.shape[0]

        # Loads the RGB and depth images.
        associations_file = self.seq_dir / "associations.txt"
        with open(associations_file, "r", encoding="utf-8") as f:
            line_iter = (line.strip().split() for line in f.readlines() if line)
            self.frames, self.depths = list(p for p in zip(*((frame, depth) for _, frame, _, depth in line_iter)))
        assert len(self.frames) == self.num_frames
        assert len(self.depths) == self.num_frames

        # Loads the camera intrinsics.
        with open(seqs_dir / "replicaCAD_info.json", "r", encoding="utf-8") as f:
            info = json.load(f)
        cam_params = info["camera"]
        self.intrinsics = intrinsics_matrix(cam_params["fx"], cam_params["fy"], cam_params["cx"], cam_params["cy"])
        self.depth_scale: float = info["depth_scale"]

        self.img_dim = img_dim
        self.random_crop = random_crop

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> PosedRGBDItem:
        depth = torch.from_numpy((np.asarray(Image.open(self.seq_dir / self.depths[index])) / self.depth_scale).copy())
        depth = depth.unsqueeze(0).to(torch.float32)
        img = torch.from_numpy(np.asarray(Image.open(self.seq_dir / self.frames[index])).copy())
        img = img.permute(2, 0, 1)[:3]  # Cut off alpha channel
        img = V.convert_image_dtype(img, torch.float32)
        pose = torch.from_numpy(self.traj[index])
        mask = depth == 0.0
        intrinsics = self.intrinsics

        if self.img_dim is not None:
            if self.random_crop:
                crop_shape = [self.img_dim, self.img_dim]
                img, depth, mask = ml.transforms.random_square_crop_multi([img, depth, mask])
                img, depth, mask = V.resize(img, crop_shape), V.resize(depth, crop_shape), V.resize(mask, crop_shape)
            else:
                img = ml.transforms.square_resize_crop(img, self.img_dim)
                depth = ml.transforms.square_resize_crop(depth, self.img_dim)
                mask = ml.transforms.square_resize_crop(mask, self.img_dim)

            # Currently haven't implemented intrinsics rescaling when cropped.
            intrinsics.fill_(-1.0)

        item = PosedRGBDItem(image=img, depth=depth, mask=mask, intrinsics=intrinsics, pose=pose)

        if ml.is_debugging():
            item.check()
        return item


if __name__ == "__main__":
    # python -m usa.tasks.datasets.impl.replica_cad
    visualize_posed_rgbd_dataset(ReplicaCADDataset("apt_3_mnp"), max_point_cloud_samples=10)
