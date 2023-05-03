import argparse
import logging
import os
from pathlib import Path
from typing import Type, cast

import ml.api as ml
import numpy as np
import open3d as o3d
from torch.utils.data.dataset import Dataset

from usa.evaluation.utils import load_clip_sdf_model_from_ckpt_and_config
from usa.models.point2emb import Point2EmbModel
from usa.planners.clip_sdf import GradientPlanner as GradientClipSDFPlanner
from usa.tasks.clip_sdf import ClipSdfTask
from usa.tasks.datasets.types import PosedRGBDItem
from usa.tasks.datasets.utils import make_point_cloud_from_dataset

logger = logging.getLogger(__name__)


def visualize_semantics(
    model: Point2EmbModel,
    task: ClipSdfTask,
    device: Type[ml.BaseDevice],
    floor_ceil_heights: tuple[float, float],
    dataset: Dataset[PosedRGBDItem],
    goals: list[str],
    artifacts_dir: Path,
    resolution: float,
) -> None:
    floor_height, ceil_height = floor_ceil_heights

    planner = GradientClipSDFPlanner(
        dataset,
        model,
        task,
        device,
        cache_dir=None,  # No caching
        floor_height=floor_height,
        ceil_height=ceil_height,
    )

    point_cloud = make_point_cloud_from_dataset(
        dataset,
        batch_size=4,
        num_workers=0,
        voxel_size=resolution,
        max_batch_points=100_000,
        max_height=ceil_height,  # Remove the ceiling.
    )

    o3d.io.write_point_cloud(str(artifacts_dir / "point_cloud.ply"), point_cloud)

    def clean_goal(goal: str) -> str:
        return goal.replace(" ", "_").replace("/", "_").lower()

    for goal in goals:
        # Gets the scores for all the points in the point cloud.
        point_cloud_xyz = [(p[0], p[1], p[2]) for p in point_cloud.points]
        scores = planner.score_locations(goal, point_cloud_xyz)
        point_cloud_scores = scores[:-2]

        # Gets the points higher than some threshold.
        top_pts = np.argsort(point_cloud_scores)
        top_pts_scores = np.array(point_cloud_scores)[top_pts]
        min_top_score = (top_pts_scores[-1] - top_pts_scores[0]) * 0.9 + top_pts_scores[0]
        top_pts = top_pts[top_pts_scores > min_top_score]
        top_pts_xyz = np.array(point_cloud_xyz)[top_pts]
        top_point_cloud_path = artifacts_dir / f"{clean_goal(goal)}_top.ply"
        top_point_cloud = o3d.geometry.PointCloud()
        top_point_cloud.points = o3d.utility.Vector3dVector(top_pts_xyz)
        top_point_cloud.paint_uniform_color([1, 0, 0])
        o3d.io.write_point_cloud(str(top_point_cloud_path), top_point_cloud)

    logger.info("Wrote point clouds to %s", artifacts_dir)


def main() -> None:
    """Entry point for the script to visualize the scene semantics.

    This script outputs a point cloud for the scene, plus a point cloud
    highlighting the points which have the highest similarity with some
    sematnic target.
    """

    ml.configure_logging(use_tqdm=True)

    parser = argparse.ArgumentParser(description="Get trajectories for a trained model.")
    parser.add_argument(
        "--floor-height",
        type=float,
        default=0.1,
        help="The height of the floor.",
        required=True,
    )
    parser.add_argument(
        "--ceil-height",
        type=float,
        default=1.0,
        help="The height of the ceiling.",
        required=True,
    )
    parser.add_argument(
        "--goals",
        type=str,
        help="The target location(s) as a list of semantic goals, separated by semicolons.",
        required=True,
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="The path to the checkpoint to use.",
        required=True,
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path to the config file.",
        required=True,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        help="The path to the directory to save artifacts to.",
        required=True,
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=5e-2,
        help="The resolution of the point cloud, in meters.",
    )
    args = parser.parse_args()

    # Gets the device (GPU, CPU, Metal...)
    os.environ["USE_FP64"] = "1"
    device = ml.AutoDevice.detect_device()

    # Loads the model from the model root.
    ckpt_path, config_path = Path(args.ckpt_path), Path(args.config_path)
    model, task = load_clip_sdf_model_from_ckpt_and_config(ckpt_path, config_path, device)

    # Gets the dataset.
    floor_ceil_heights = float(args.floor_height), float(args.ceil_height)
    dataset = task.get_dataset("train")

    # Gets the resolution, goal locations and artifacts directory from arguments.
    resolution = float(args.resolution)
    goals = cast(list[str], args.goals.split(";"))
    artifacts_dir = Path(args.artifacts_dir)

    visualize_semantics(model, task, device, floor_ceil_heights, dataset, goals, artifacts_dir, resolution)


if __name__ == "__main__":
    main()
