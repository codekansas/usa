import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import ml.api as ml
import numpy as np
import torch
import torchvision.transforms.functional as V
import tqdm
from quaternion import as_rotation_matrix, quaternion
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_url

from usa.tasks.datasets.types import PosedRGBDItem
from usa.tasks.datasets.utils import visualize_posed_rgbd_dataset

# Pre-recorded clips used in paper.
LAB_CLIP_URL = "https://drive.google.com/file/d/1-bJ08bSnMFqz82OolCSbiyY430UstNzL/view?usp=share_link"
KITCHEN_CLIP_URL = "https://drive.google.com/file/d/133D0ydVu2I3ddS69qdoJ-5HtcRgyrvgf/view?usp=share_link"
CHESS_CLIP_URL = "https://drive.google.com/file/d/1ElpXCMclP9xocU1Kd5OyrtkO3UNSbzf8/view?usp=share_link"


def iter_raw(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "rb") as f:
        while True:
            try:
                yield pkl.load(f)
            except EOFError:
                break


@dataclass
class OccupancyMap:
    image: np.ndarray
    shape: Tuple[int, int]
    origin: Tuple[float, float]
    resolution: float


@dataclass
class Data:
    color_imgs: np.ndarray
    depth_imgs: np.ndarray
    poses: np.ndarray
    intrinsics: np.ndarray
    occupancy_maps: List[OccupancyMap]


def load_data(path: Path) -> Data:
    color_imgs: List[np.ndarray] = []
    depth_imgs: List[np.ndarray] = []
    poses: List[np.ndarray] = []
    intrinsics: List[np.ndarray] = []
    occupancy_maps: List[OccupancyMap] = []
    for data in tqdm.tqdm(iter_raw(path), disable=ml.is_distributed(), desc="Loading data"):
        color_img = data["color"]["image"]
        depth_img = data["depth"]["image"]

        # Gets the occupancy map for the current frame.
        occ_map_data = data["occupancy_map"]
        occupancy_map = OccupancyMap(
            image=occ_map_data["image"],
            shape=occ_map_data["shape"],
            origin=occ_map_data["origin"],
            resolution=occ_map_data["resolution"],
        )
        occupancy_maps.append(occupancy_map)

        # color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # Adds channels to the depth image.
        color_imgs.append(color_img)
        depth_imgs.append(depth_img)

        # Get the scaled intrinsics.
        color_ds = data["color"]["downsample_factor"]
        unscaled_intr = np.array(data["color"]["K"]).reshape(3, 3)
        intr = np.eye(3, 3)
        intr[0, 0] = unscaled_intr[0, 0] / color_ds  # fx
        intr[1, 1] = unscaled_intr[1, 1] / color_ds  # fy
        intr[0, 2] = unscaled_intr[0, 2] / color_ds  # cx
        intr[1, 2] = unscaled_intr[1, 2] / color_ds  # cy
        # intr = unscaled_intr
        intrinsics.append(intr)

        # R and P are only used for stereo cameras.
        rot = np.array(data["color"]["R"]).reshape(3, 3)
        proj = np.array(data["color"]["P"]).reshape(3, 4)
        assert (rot == np.eye(3)).all()
        assert (proj[:3, :3] == unscaled_intr).all()
        assert (proj[:3, 3] == 0).all()

        # Builds the pose from the quaternions and positions.
        pose = np.eye(4, dtype=np.float64)

        qx, qy, qz, qw = data["color"]["rot"]
        xyz = data["color"]["xyz"]

        pose[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        pose[:3, 3] = xyz

        poses.append(pose)

    return Data(
        color_imgs=np.stack(color_imgs, axis=0),
        depth_imgs=np.stack(depth_imgs, axis=0),
        poses=np.stack(poses, axis=0),
        intrinsics=np.stack(intrinsics, axis=0),
        occupancy_maps=occupancy_maps,
    )


def load_pose(path: Path) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    inds = np.genfromtxt(path, delimiter=",", skip_header=True, usecols=[2, 3], dtype=np.int64)
    poses = np.genfromtxt(path, delimiter=",", skip_header=True, usecols=[4, 5, 6, 7, 8, 9, 10])
    mat = np.eye(4, dtype=np.float64)[None, :, :].repeat(poses.shape[0], axis=0)
    mat[:, :3, :3] = np.stack([as_rotation_matrix(quaternion(qw, qx, qy, qz)) for qw, qx, qy, qz in poses[:, 3:]])
    mat[:, :3, 3] = poses[:, :3]
    ids = list((a, b) for a, b in zip(inds[:, 0].tolist(), inds[:, 1].tolist()))
    return mat, ids


class StretchDataset(Dataset[PosedRGBDItem]):
    def __init__(
        self,
        path: str | Path,
        min_depth: float = 0.3,
        max_depth: float = 3.0,
    ) -> None:
        """Dataset for clips recorded from the Stretch robot.

        The default minimum and maximum depths are taken from here:
            https://www.phase1vision.com/cameras/intel-realsense/stereo-depth/depth-d435i

        For new datasets, it's useful to visualize the point clouds to
        ensure that the visualizations look correct (i.e., not too much noise
        from depth estimates).

        Args:
            path: Path to the dataset (a Pickle file).
            min_depth: Minimum depth in meters.
            max_depth: Maximum depth in meters.
        """

        super().__init__()

        self.min_depth = min_depth
        self.max_depth = max_depth

        # Loads the data from the Pickle file.
        data = load_data(Path(path))
        self.colors = data.color_imgs
        self.depths = data.depth_imgs
        self.poses = data.poses
        self.intrinsics = data.intrinsics
        self.occupancy_maps = data.occupancy_maps

    def __getitem__(self, index: int) -> PosedRGBDItem:
        color, depth, pose, intr = self.colors[index], self.depths[index], self.poses[index], self.intrinsics[index]

        # Converts depth to meters with correct shape.
        depth_fp = torch.from_numpy(depth.astype(np.float32) / 1000.0)
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


class StretchDatasetDownload(StretchDataset):
    def __init__(self, url: str, key: str) -> None:
        filename = f"{key}.pkl"
        clip_path = ml.get_data_dir() / "stretch" / filename
        if not clip_path.exists():
            download_url(url, clip_path.parent, filename=filename)
        super().__init__(clip_path)


def lab_stretch_dataset() -> StretchDataset:
    return StretchDatasetDownload(LAB_CLIP_URL, "lab")


def kitchen_stretch_dataset() -> StretchDataset:
    return StretchDatasetDownload(KITCHEN_CLIP_URL, "kitchen")


def chess_stretch_dataset() -> StretchDataset:
    return StretchDatasetDownload(CHESS_CLIP_URL, "chess")


if __name__ == "__main__":
    # python -m usa.tasks.datasets.impl.stretch
    visualize_posed_rgbd_dataset(
        lab_stretch_dataset(),
        make_video=False,
        make_point_cloud=True,
    )
