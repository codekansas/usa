import logging
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import ml.api as ml
import numpy as np
import torch
import torchvision.transforms.functional as V
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_url

from usa.tasks.datasets.types import PosedRGBDItem
from usa.tasks.datasets.utils import visualize_posed_rgbd_dataset

logger = logging.getLogger(__name__)

CHRIS_LAB_URL = "https://github.com/codekansas/usa/releases/download/v0.0.2/chris_lab.pkl"


@dataclass
class Data:
    color_imgs: np.ndarray
    depth_imgs: np.ndarray
    poses: np.ndarray
    intrinsics: np.ndarray


def load_data(pkl_path: Path) -> Data:
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    return Data(
        color_imgs=np.array(data["orig_rgb"]),
        depth_imgs=np.array(data["orig_depth"]),
        poses=np.stack(data["poses"]),
        intrinsics=np.array(data["K"]),
    )


class HomeRobotDataset(Dataset[PosedRGBDItem]):
    def __init__(
        self,
        path: str | Path,
        min_depth: float = 0.3,
        max_depth: float = 3.0,
    ) -> None:
        """Dataset for clips recorded from the Home Robot repository.

        Args:
            path: Path to the dataset.
            min_depth: Minimum depth to use.
            max_depth: Maximum depth to use.
        """

        super().__init__()

        self.path = Path(path)
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Loads the data from the Pickle file.
        data = load_data(self.path)
        self.colors = data.color_imgs
        self.depths = data.depth_imgs
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
            mask=(depth_fp < self.min_depth) | (depth_fp > self.max_depth),
            intrinsics=torch.from_numpy(intr),
            pose=torch.from_numpy(pose),
        )

        item.check()

        return item

    def __len__(self) -> int:
        return self.colors.shape[0]


class HomeRobotDatasetDownload(HomeRobotDataset):
    def __init__(self, url: str, key: str) -> None:
        filename = f"{key}.pkl"
        clip_path = ml.get_data_dir() / "home_robot" / filename
        if not clip_path.exists():
            download_url(url, clip_path.parent, filename=filename)
        super().__init__(clip_path)


def chris_lab_home_robot_dataset() -> HomeRobotDataset:
    return HomeRobotDatasetDownload(CHRIS_LAB_URL, "chris_lab")


if __name__ == "__main__":
    # python -m usa.tasks.datasets.impl.home_robot
    ml.configure_logging()
    visualize_posed_rgbd_dataset(
        chris_lab_home_robot_dataset(),
        make_video=True,
        make_point_cloud=True,
    )
