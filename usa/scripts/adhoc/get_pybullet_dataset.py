import argparse
import logging
import os
import pickle as pkl
import zipfile
from pathlib import Path
from typing import Iterator

import imageio
import matplotlib.pyplot as plt
import ml.api as ml
import numpy as np
import pybullet as pb
import requests

from usa.tasks.datasets.pybullet import PyBulletDataset
from usa.tasks.datasets.utils import visualize_posed_rgbd_dataset

logger = logging.getLogger(__name__)


def capture_frame(
    camera_xyz: tuple[float, float, float] = (-5.0, 0.0, 1.477),
    camera_ypr: tuple[float, float, float] = (90.0, -10.0, 0.0),
    camera_planes: tuple[float, float] = (0.01, 10.0),
    pixel_dims: tuple[int, int] = (500, 300),
    camera_fov: float = 80.0,
) -> tuple[np.ndarray, ...]:
    """Captures a single frame, returning RGB and depth information.

    Args:
        camera_xyz: The XYZ coordinates of the camera
        camera_ypr: The yaw, pitch and roll of the camera
        camera_planes: The minimum and maximum rendering distances
        pixel_dims: The shape of the output image, as (W, H)
        camera_fov: The camera field of view

    Returns:
        The RGB image with shape (H, W, 3), the depth image with shape (H, W),
        the intrinsics matrix with shape (3, 3), and the pose matrix with
        shape (4, 4).
    """

    x, y, z = camera_xyz
    yaw, pitch, roll = camera_ypr
    near_plane, far_plane = camera_planes
    pixel_width, pixel_height = pixel_dims

    # Computes the view and projection matrices.
    view_mat = pb.computeViewMatrixFromYawPitchRoll(camera_xyz, near_plane, yaw, pitch, roll, 2)
    aspect = pixel_width / pixel_height
    proj_mat = pb.computeProjectionMatrixFOV(camera_fov, aspect, near_plane, far_plane)

    # Captures the camera image.
    img_arr = pb.getCameraImage(
        width=pixel_width,
        height=pixel_height,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
    )
    img_width, img_height, rgb, depth, info = img_arr

    # Reshapes arrays to expected output shape.
    rgb_arr = np.reshape(rgb, (img_height, img_width, 4))[..., :3]
    depth_arr = np.reshape(depth, (img_height, img_width))

    # Converts depth to true depth.
    depth_arr = far_plane * near_plane / (far_plane - (far_plane - near_plane) * depth_arr)

    # Gets camera intrinsics matrix.
    cx = pixel_width / 2
    cy = pixel_height / 2
    fov_rad = np.deg2rad(camera_fov)
    fx = cx / np.tan(fov_rad / 2)
    fy = cy / np.tan(fov_rad / 2)

    """
    proj_mat_np = np.array(proj_mat, dtype=np.float64).reshape(4, 4, order="F")
    fx = proj_mat_np[0, 0]
    fy = proj_mat_np[1, 1]
    cx = proj_mat_np[0, 2]
    cy = proj_mat_np[1, 2]
    """

    intr = np.eye(3)
    intr[0, 0] = fx
    intr[1, 1] = fy
    intr[0, 2] = cx
    intr[1, 2] = cy

    # Gets poses from view matrix.
    pose = np.linalg.inv(np.array(view_mat, dtype=np.float64).reshape(4, 4, order="F"))
    affine_mat = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )
    pose = pose @ affine_mat

    return rgb_arr, depth_arr, intr, pose


def capture_sim(capture_every: int = 1) -> Iterator[tuple[np.ndarray, ...]]:
    # xyz, ypr = (-5.0, 0.0, 1.477), (90.0, -10.0, 0.0)
    for i in range(90):
        degs = i * 4
        rads = np.deg2rad(degs)
        dist = 3.0
        xyz = (dist * np.cos(rads), dist * np.sin(rads), 1.477)
        # xyz = (0.0, 0.0, 1.477 + np.sin(rads) * 0.5)
        # xyz = (0.0, 0.0, 1.477)
        # ypr = (degs, -10.0 + np.sin(rads) * 10.0, 0.0)
        ypr = (degs + 90.0, -10.0, 0.0)
        if i % capture_every == 0:
            yield capture_frame(xyz, ypr)


def write_gif(
    frames: Iterator[tuple[np.ndarray, ...]],
    out_file: str | Path,
    pkl_file: str | Path,
    *,
    fps: int = 30,
) -> None:
    rgb, depth, mask, poses, intrinsics = [], [], [], [], []

    writer = imageio.get_writer(str(out_file), mode="I", fps=fps)
    for rgb_frame, depth_frame, intr, pose in frames:
        # Adds to the lists.
        rgb.append(rgb_frame)
        depth.append(depth_frame)
        mask.append(depth_frame > 7.0)
        # mask.append(np.zeros_like(depth_frame, dtype=np.bool_))
        poses.append(pose)
        intrinsics.append(intr)

        # Adds the image to the GIF.
        depth_normalized = (depth_frame - np.min(depth_frame)) / (np.max(depth_frame) - np.min(depth_frame) + 1e-3)
        depth_colorized = (plt.cm.jet(depth_normalized)[..., :3] * 255).astype(np.uint8)
        frame = np.concatenate([rgb_frame, depth_colorized], axis=0)
        writer.append_data(frame)

    # Saves the pickle file.
    data = {
        "rgb": np.stack(rgb),
        "depth": np.stack(depth),
        "mask": np.stack(mask),
        "poses": np.stack(poses),
        "intrinsics": np.stack(intrinsics),
    }
    with open(pkl_file, "wb") as f:
        pkl.dump(data, f)

    # Closes the writer.
    writer.close()


def iter_frames() -> Iterator[tuple[np.ndarray, ...]]:
    reset_simulation()
    yield from capture_sim()


def reset_simulation() -> Path:
    data_root = Path("data")
    data_root.mkdir(exist_ok=True)

    # Downloads the dataset, if it is not already downloaded.
    if not (data_root / "04_pybullet_data").exists():
        r = requests.get(
            "https://github.com/codekansas/usa/releases/download/v0.0.2/04_pybullet_data.zip", allow_redirects=True
        )
        with open(data_root / "04_pybullet_data.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(data_root / "04_pybullet_data.zip", "r") as zip_ref:
            zip_ref.extractall(data_root)

    # Loads the URDFs into PyBullet.
    pb.setAdditionalSearchPath(str(data_root / "04_pybullet_data"))
    use_fixed_base = True
    pb.setGravity(0, 0, -9.81)
    pb.resetSimulation()
    pb.setGravity(0, 0, -9.81)
    pb.setPhysicsEngineParameter(enableConeFriction=0)

    pb.loadURDF(
        "floor.urdf",
        useFixedBase=use_fixed_base,
    )

    pb.loadURDF(
        "kitchen_part_right_gen_convex.urdf",
        (0.0, 0, 1.477),
        useFixedBase=use_fixed_base,
    )

    pb.loadURDF(
        "table.urdf",
        (2.5, 0, 0),
        pb.getQuaternionFromEuler((0, 0, 1.57)),
        useFixedBase=use_fixed_base,
    )

    return data_root


def set_environ_vars() -> None:
    # These environment variables control where training and eval logs are written.
    # You can set these in your shell profile as well.
    os.environ["RUN_DIR"] = "runs"
    os.environ["EVAL_RUN_DIR"] = "eval_runs"
    os.environ["MODEL_DIR"] = "models"
    os.environ["DATA_DIR"] = "data"

    # This is used to set a constant Tensorboard port.
    os.environ["TENSORBOARD_PORT"] = str(8989)

    # Useful for debugging.
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main() -> None:
    """Builds a PyBullet dataset.

    Usage:
        python -m usa.scripts.adhoc.get_pybullet_dataset

    This will write PyBullet data artifacts to a `data` directory in whatever
    directory you run this script from.
    """

    ml.configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--point-cloud-batches", type=int, default=2)
    parser.add_argument("-s", "--point-cloud-stride", type=int, default=5)
    parser.add_argument("-v", "--make-video", default=False, action="store_true")
    args = parser.parse_args()

    set_environ_vars()
    pb.connect(pb.DIRECT)

    data_root = reset_simulation()

    # Writes the GIF and data pickle file.
    video_path = data_root / "video.gif"
    pkl_path = data_root / "04_recorded_clip.pkl"
    write_gif(iter_frames(), video_path, pkl_path)

    # Loads the
    dataset = PyBulletDataset(path=pkl_path)

    # Creates a point cloud from the dataset.
    visualize_posed_rgbd_dataset(
        dataset,
        make_video=False,
        make_point_cloud=True,
        max_point_cloud_samples=args.point_cloud_batches,
        point_cloud_sample_stride=args.point_cloud_stride,
        output_dir=data_root,
    )

    logger.info("Wrote point cloud to %s", data_root / "point_cloud.ply")


if __name__ == "__main__":
    main()
