import itertools
import logging
import os
import time
from pathlib import Path
from typing import Iterator

import ml.api as ml
import numpy as np
import open3d as o3d
import torch
import tqdm
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

from usa.tasks.datasets.types import PosedRGBDItem

logger = logging.getLogger(__name__)


def aminmax(x: Tensor, dim: int | None = None) -> tuple[Tensor, Tensor]:
    if x.device.type == "mps":
        return x.min(dim=dim)[0], x.max(dim=dim)[0]
    xmin, xmax = torch.aminmax(x, dim=dim)
    return xmin, xmax


def test_dataset(ds: Dataset | IterableDataset | DataLoader, max_samples: int = 3) -> None:
    """Iterates through a dataset.

    Args:
        ds: The dataset to iterate through
        max_samples: Maximum number of samples to loop through
    """

    ml.configure_logging(use_tqdm=True)
    start_time = time.time()

    if isinstance(ds, (IterableDataset, DataLoader)):
        logger.info("Iterating samples in %s", "dataloader" if isinstance(ds, DataLoader) else "dataset")
        for i, _ in enumerate(itertools.islice(ds, max_samples)):
            if i % 10 == 0:
                logger.info("Sample %d in %.2g seconds", i, time.time() - start_time)
    else:
        samples = len(ds)  # type: ignore
        logger.info("Dataset has %d items", samples)
        for i in tqdm.trange(min(samples, max_samples)):
            _ = ds[i]
            if i % 10 == 0:
                logger.info("Sample %d in %.2g seconds", i, time.time() - start_time)


def get_inv_pose(pose: Tensor) -> Tensor:
    # return pose.double().inverse().to(pose)
    rot, trans = pose[..., :3, :3], pose[..., :3, 3]
    inv_pose = torch.cat((rot.transpose(-1, -2), -torch.einsum("nab,na->nb", rot, trans).unsqueeze(-1)), dim=-1)
    return torch.cat((inv_pose, inv_pose.new_tensor((0, 0, 0, 1)).expand_as(inv_pose[..., :1, :])), dim=-2)


def get_inv_intrinsics(intrinsics: Tensor) -> Tensor:
    # return intrinsics.double().inverse().to(intrinsics)
    fx, fy, ppx, ppy = intrinsics[..., 0, 0], intrinsics[..., 1, 1], intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    inv_intrinsics = torch.zeros_like(intrinsics)
    inv_intrinsics[..., 0, 0] = 1.0 / fx
    inv_intrinsics[..., 1, 1] = 1.0 / fy
    inv_intrinsics[..., 0, 2] = -ppx / fx
    inv_intrinsics[..., 1, 2] = -ppy / fy
    inv_intrinsics[..., 2, 2] = 1.0
    return inv_intrinsics


def apply_pose(xyz: Tensor, pose: Tensor) -> Tensor:
    return (torch.einsum("na,nba->nb", xyz.to(pose), pose[..., :3, :3]) + pose[..., :3, 3]).to(xyz)


def apply_intrinsics(xyz: Tensor, intrinsics: Tensor) -> Tensor:
    return torch.einsum("na,nba->nb", xyz.to(intrinsics), intrinsics)[..., :2]


def apply_inv_intrinsics(xy: Tensor, intrinsics: Tensor) -> Tensor:
    inv_intrinsics = get_inv_intrinsics(intrinsics)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)
    return torch.einsum("na,nba->nb", xyz.to(inv_intrinsics), inv_intrinsics)


def get_xy_pixel_from_xyz(xyz: Tensor, pose: Tensor, intrinsics: Tensor) -> Tensor:
    """Gets the XY pixel coordinate from the image XYZ coordinate.

    Args:
        xyz: The XYZ coordinates in the global frame, with shape (N, 3)
        pose: The inverse pose, with shape (N, 4, 4)
        intrinsics: The intrinsics, with shape (N, 3, 3)

    Returns:
        The XY coordinates of the pixel in the camera frame, with shape (N, 2)
    """

    xyz = apply_pose(xyz, get_inv_pose(pose))
    xyz = xyz / xyz[..., 2:]
    xy = apply_intrinsics(xyz, intrinsics)
    return xy


def get_xyz_coordinates_from_xy(depth: Tensor, xy: Tensor, pose: Tensor, intrinsics: Tensor) -> Tensor:
    """Gets XYZ coordinates from image XY coordinates.

    Args:
        depth: The depths for each XY coordinate, with shape (N)
        xy: The XY coordinates, with shape (N, 2)
        pose: The pose, with shape (N, 4, 4)
        intrinsics: The intrinsics, with shape (N, 3, 3)

    Returns:
        The XYZ coordinates in the global frame, with shape (N, 3)
    """

    xyz = apply_inv_intrinsics(xy, intrinsics)
    xyz = xyz * depth[:, None]
    xyz = apply_pose(xyz, pose)
    return xyz


def get_xyz_coordinates(depth: Tensor, mask: Tensor, pose: Tensor, intrinsics: Tensor) -> Tensor:
    """Returns the XYZ coordinates for a posed RGBD image.

    Args:
        depth: The depth tensor, with shape (B, 1, H, W)
        mask: The mask tensor, with the same shape as the depth tensor,
            where True means that the point should be masked (not included)
        intrinsics: The inverse intrinsics, with shape (B, 3, 3)
        pose: The poses, with shape (B, 4, 4)

    Returns:
        XYZ coordinates, with shape (N, 3) where N is the number of points in
        the depth image which are unmasked
    """

    bsz, _, height, width = depth.shape
    flipped_mask = ~mask

    # Associates poses and intrinsics with XYZ coordinates.
    batch_inds = torch.arange(bsz, device=mask.device)
    batch_inds = batch_inds[:, None, None, None].expand_as(mask)[~mask]
    intrinsics = intrinsics[batch_inds]
    pose = pose[batch_inds]

    # Gets the depths for each coordinate.
    depth = depth[flipped_mask]

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(width, device=depth.device),
        torch.arange(height, device=depth.device),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1)[None, :, :].repeat_interleave(bsz, dim=0)
    xy = xy[flipped_mask.squeeze(1)]

    return get_xyz_coordinates_from_xy(depth, xy, pose, intrinsics)


def get_nearest_xyz(all_xyz: Tensor, xyz: Tensor) -> tuple[Tensor, Tensor]:
    """Gets nearest neighbor coordinates.

    Args:
        all_xyz: The points to compare to, with shape (B, 3)
        xyz: The points to get nearest neighbors to, with shape (N, 3)

    Returns:
        The nearest neighbors from `all_xyz`, with shape (N, 3), along with
        the indices of the nearest points in `all_xyz`, with shape (N)
    """

    with torch.no_grad():
        nearest = torch.cdist(xyz[None], all_xyz[None]).squeeze(0).argmin(1)
        return all_xyz[nearest], nearest

        # chunks = all_xyz.split(self.config.chunk_size)
        # nearest_points = torch.stack([torch.cdist(chunk[None], xyz[None])[0].argmin(dim=0) for chunk in chunks])
        # sub_xyz = all_xyz[nearest_points.flatten()].unflatten(0, nearest_points.shape)
        # sub_xyz_inds = torch.norm(sub_xyz - xyz[None], dim=-1).argmin(dim=0)
        # return torch.gather(sub_xyz, 0, sub_xyz_inds[None, :, None].repeat_interleave(3, -1)).squeeze(0)


def make_video_from_dataset(
    ds: Dataset[PosedRGBDItem],
    save_path: str | Path,
    max_samples: int | None = None,
    rotate: bool = False,
    concat_horizontal: bool = False,
) -> None:
    """Makes a video from an R3D dataset.

    Args:
        ds: The dataset to make a video from
        save_path: Where to save the created video
        max_samples: The maximum number of samples to use, or None to use all
        rotate: Whether to rotate the camera 90 degrees
        concat_horizontal: Whether to concatenate the images horizontally
    """

    def iter_frames() -> Iterator[np.ndarray]:
        i = 0
        for item in tqdm.tqdm(ds, desc="Processing video"):
            depth = (item.depth - item.depth.min()) / (item.depth.max() - item.depth.min())
            reg_depth = (255 * depth).to(torch.uint8)
            # inv_depth = (255 / (32 * depth + 1)).to(torch.uint8)
            image = (item.image * 255).to(torch.uint8)
            # mask = item.mask.to(torch.uint8) * 255

            if rotate:
                reg_depth = reg_depth.permute(0, 2, 1).flip(2)
                # inv_depth = inv_depth.permute(0, 2, 1).flip(2)
                image = image.permute(0, 2, 1).flip(2)
                # mask = mask.permute(0, 2, 1).flip(2)

            # Creates the image; depth image is on the bottom,
            # RGB image and mask are on the top.
            concatted = torch.cat(
                [
                    image,
                    reg_depth.repeat_interleave(3, dim=0),
                ],
                dim=2 if concat_horizontal else 1,
            )

            # C, H, W -> H, W, C
            yield concatted.permute(1, 2, 0).numpy()

            i += 1
            if max_samples is not None and i >= max_samples:
                break

    ml.WRITERS["ffmpeg"](iter_frames(), save_path)


def make_point_cloud_from_dataset(
    ds: Dataset[PosedRGBDItem],
    batch_size: int = 4,
    num_workers: int = 0,
    voxel_size: float = 5e-2,
    max_batch_points: int = 100_000,
    max_samples: int | None = None,
    max_height: float | None = None,
    stride: int = 1,
) -> o3d.geometry.PointCloud:
    """Makes a point cloud from an R3D dataset.

    Args:
        ds: The dataset to make a point cloud from
        batch_size: Batch size for iterating frames
        num_workers: Number of workers for dataloader
        voxel_size: The voxel size to use for downsampling the point cloud
        max_batch_points: Maximum number of points in a batch, to
            avoid overloading Open3d
        max_samples: Maximum number of samples to use, or None to use all
        max_height: Maximum height to use, or None to use all
        stride: Stride to use when iterating frames

    Returns:
        The point cloud
    """

    device = ml.AutoDevice.detect_device()
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ml.collate,
    )

    pf = device.get_prefetcher(dl)
    total = (len(ds) + batch_size - 1) // batch_size  # type: ignore

    # Saves the complete point cloud.
    final_pcd = o3d.geometry.PointCloud()

    i, j = 0, 0

    with torch.inference_mode():
        for item in tqdm.tqdm(pf, desc="Processing point cloud", total=total):
            j += 1
            if j < stride:
                continue
            j = 0

            image, depth, mask, intrinsics, pose = item

            xyz = get_xyz_coordinates(depth, mask, pose, intrinsics)
            colors = image.permute(0, 2, 3, 1)[~mask.squeeze(1)]

            assert colors.shape[0] == xyz.shape[0]

            # Filters out points above a certain height.
            if max_height is not None:
                max_height_mask = xyz[:, -1] < max_height
                xyz = xyz[max_height_mask]
                colors = colors[max_height_mask]

            # Downsampling to avoid super large point cloud.
            if xyz.shape[0] > max_batch_points:
                inds = torch.randperm(xyz.shape[0], device=xyz.device)[:max_batch_points]
                xyz = xyz[inds]
                colors = colors[inds]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz.cpu())
            pcd.colors = o3d.utility.Vector3dVector(colors.cpu())
            final_pcd = final_pcd + pcd

            # Voxel downsample to avoid storing too much data.
            final_pcd = final_pcd.voxel_down_sample(voxel_size=voxel_size)

            i += 1
            if max_samples is not None and i >= max_samples:
                break

    return final_pcd


def visualize_posed_rgbd_dataset(
    ds: Dataset[PosedRGBDItem],
    make_video: bool = False,
    make_point_cloud: bool = True,
    output_dir: str | Path = "out",
    max_video_samples: int | None = None,
    max_point_cloud_samples: int | None = None,
    point_cloud_sample_stride: int = 1,
    rotate: bool = False,
    concat_horizontal: bool = False,
) -> None:
    """Provides an ad-hoc test script for the ReplicaCAD dataset.

    Usage:
        python -m ml.tasks.datasets.impl.stretch

    Args:
        ds: The dataset to visualize
        make_video: If set, make a video of the clip
        make_point_cloud: If set, make a point cloud from the clip
        output_dir: Where to save the video and point cloud
        max_video_samples: The maximum number of samples to use for the video,
            or None to use all
        max_point_cloud_samples: Maximum number of samples to use for
            the point cloud, or None to use all
        point_cloud_sample_stride: The stride to use for the point cloud
        rotate: If set, rotate the video 90 degrees
        concat_horizontal: If set, concatenate the video horizontally
    """

    # Disabling Metal because it doesn't support fp64.
    os.environ["DISABLE_METAL"] = "1"

    ml.Debugging.set(True)  # Always log debug information here.
    ml.configure_logging()

    # Gets the output directory.
    (output_dir_path := Path(output_dir)).mkdir(exist_ok=True, parents=True)
    video_path = (output_dir_path / "video.mp4").resolve()
    point_cloud_path = (output_dir_path / "point_cloud.ply").resolve()

    # Generates a video from the dataset.
    if make_video:
        make_video_from_dataset(
            ds,
            video_path,
            max_samples=max_video_samples,
            rotate=rotate,
            concat_horizontal=concat_horizontal,
        )

    # Generates a point cloud from the dataset.
    if make_point_cloud:
        point_cloud = make_point_cloud_from_dataset(
            ds,
            max_samples=max_point_cloud_samples,
            stride=point_cloud_sample_stride,
        )
        o3d.io.write_point_cloud(str(point_cloud_path), point_cloud)

    # Finally, print the absolute paths to the saved files.
    if make_video:
        logger.info("Saved video to %s", video_path)
    if make_point_cloud:
        logger.info("Saved point cloud to %s", point_cloud_path)
