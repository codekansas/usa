import logging
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as V
from torch.utils.data.dataset import Dataset

from usa.tasks.datasets.types import PosedRGBDItem

logger = logging.getLogger(__name__)


@dataclass
class Data:
    color_imgs: np.ndarray
    depth_imgs: np.ndarray
    masks: np.ndarray
    poses: np.ndarray
    intrinsics: np.ndarray


def load_data(pkl_path: Path) -> Data:
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    return Data(
        color_imgs=np.array(data["rgb"]),
        depth_imgs=np.array(data["depth"]),
        masks=np.array(data["mask"]),
        poses=np.stack(data["poses"]),
        intrinsics=np.array(data["intrinsics"]),
    )


class PyBulletDataset(Dataset[PosedRGBDItem]):
    def __init__(self, path: str | Path) -> None:
        """Dataset for clips recorded from the Home Robot repository.

        Args:
            path: Path to the dataset.
        """

        super().__init__()

        self.path = Path(path)

        # Loads the data from the Pickle file.
        data = load_data(self.path)
        self.colors = data.color_imgs
        self.depths = data.depth_imgs
        self.masks = data.masks
        self.poses = data.poses
        self.intrinsics = data.intrinsics

    def __getitem__(self, index: int) -> PosedRGBDItem:
        color, depth, pose, intr = self.colors[index], self.depths[index], self.poses[index], self.intrinsics[index]

        # Converts depth to meters with correct shape.
        # depth_fp = torch.from_numpy(depth.astype(np.float32) / 1000.0)
        depth_fp = torch.from_numpy(depth.astype(np.float32))
        depth_fp = depth_fp.unsqueeze(0)

        # Converts image to floating point with correct shape.
        image = torch.from_numpy(color)
        image = V.convert_image_dtype(image, torch.float32)
        image = image.permute(2, 0, 1)

        item = PosedRGBDItem(
            image=image,
            depth=depth_fp,
            mask=torch.from_numpy(self.masks[index]),
            intrinsics=torch.from_numpy(intr),
            pose=torch.from_numpy(pose),
        )

        item.check()

        return item

    def __len__(self) -> int:
        return self.colors.shape[0]
