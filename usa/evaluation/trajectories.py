import argparse
import logging
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Type, cast, get_args

import numpy as np
import open3d as o3d
from ml.trainers.mixins.device.auto import AutoDevice
from ml.trainers.mixins.device.base import BaseDevice
from ml.utils.logging import configure_logging
from torch.utils.data.dataset import Dataset

from usa.evaluation.utils import load_clip_sdf_model_from_ckpt_and_config, plot_paths
from usa.models.point2emb import Point2EmbModel
from usa.planners.base import Planner
from usa.planners.clip_sdf import (
    AStarPlanner as AStarClipSDFPlanner,
    GradientPlanner as GradientClipSDFPlanner,
)
from usa.tasks.clip_sdf import ClipSdfTask
from usa.tasks.datasets.types import PosedRGBDItem
from usa.tasks.datasets.utils import make_point_cloud_from_dataset

logger = logging.getLogger(__name__)

PlannerType = Literal["a_star", "gradient"]


def get_planner(
    model: Point2EmbModel,
    task: ClipSdfTask,
    planner_key: PlannerType,
    floor_ceil_heights: Tuple[float, float],
    device: Type[BaseDevice],
    dataset: Optional[Dataset[PosedRGBDItem]] = None,
) -> Planner:
    if dataset is None:
        dataset = cast(Dataset[PosedRGBDItem], task.get_dataset("train"))
    floor_height, ceil_height = floor_ceil_heights

    if planner_key == "a_star":
        return AStarClipSDFPlanner(
            dataset,
            model,
            task,
            device,
            "euclidean",
            resolution=0.1,
            cache_dir=None,  # No caching
            floor_height=floor_height,
            ceil_height=ceil_height,
        )

    if planner_key == "gradient":
        return GradientClipSDFPlanner(
            dataset,
            model,
            task,
            device,
            cache_dir=None,  # No caching
            floor_height=floor_height,
            ceil_height=ceil_height,
        )

    raise NotImplementedError(f"Planner {planner_key} is not implemented.")


def print_trajectories(
    start_xy: Tuple[float, float],
    goals: List[str],
    planner: Planner,
    floor_ceil_heights: Tuple[float, float],
    artifacts_dir: Optional[Path] = None,
    dataset: Optional[Dataset[PosedRGBDItem]] = None,
    save_goals: bool = True,
) -> None:
    start_x, start_y = start_xy
    if not planner.is_valid_starting_point((start_x, start_y)):
        logger.warning("Starting point %s is not valid!", start_xy)

    all_trajectories: List[List[Tuple[float, float]]] = []
    for goal in goals:
        trajectory = planner.plan(start_xy, end_goal=goal)
        start_xy = trajectory[-1]
        all_trajectories.append(trajectory)
        logger.info("Goal: %s", goal)
        logger.info("Trajectory: %s", trajectory)

    if artifacts_dir is not None:
        # Saves the trajectories to a file.
        with open(artifacts_dir / "trajectories.txt", "w", encoding="utf-8") as f:
            for goal, trajectory in zip(goals, all_trajectories):
                trajectory_str = " ".join(f"{x},{y}" for x, y in trajectory)
                f.write(f"{goal}\t{trajectory_str}\n")

        # Plots the trajectories on top of the map.
        planner_map = planner.get_map()
        plot_paths(planner_map, all_trajectories, artifacts_dir / "trajectories.png")

        # Gets the point cloud, overlays the trajectories on top of it, and saves it.
        if dataset is not None:
            minh, maxh = floor_ceil_heights
            halfh = (maxh + minh) / 2

            point_cloud = make_point_cloud_from_dataset(
                dataset,
                batch_size=4,
                num_workers=0,
                voxel_size=planner_map.resolution,
                max_batch_points=100_000,
                max_height=maxh,  # Remove the ceiling.
            )

            if save_goals:
                all_goal_pts = []
                for goal, trajectory in zip(goals, all_trajectories):
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

                    # Find the closest point in the top 10% to the end of the trahjectory
                    xys = top_pts_xyz[:, :2]
                    dists = xys - trajectory[-1]
                    dists = np.linalg.norm(dists, axis=-1)
                    goal_idx = np.argmin(dists)
                    goal_xy = xys[goal_idx]
                    goal_point_cloud = o3d.geometry.PointCloud()
                    goal_point_cloud.points = o3d.utility.Vector3dVector([(goal_xy[0], goal_xy[1], halfh)])
                    goal_point_cloud.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
                    point_cloud += goal_point_cloud
                    all_goal_pts.append(top_pts_xyz[goal_idx, :3])

                # Write out the goal points as a part of the plan - so that we can look at them
                with open(artifacts_dir / "goals.txt", "w", encoding="utf-8") as f:
                    for goal, goal_pt in zip(goals, all_goal_pts):
                        x, y, z = goal_pt[0], goal_pt[1], goal_pt[2]
                        goal_pt_str = f"{x},{y},{z}"
                        f.write(f"{goal}\t{goal_pt_str}\n")

            # Adds the trajectories.
            flat_trajectories = [point for trajectory in all_trajectories for point in trajectory]
            traj_point_cloud = o3d.geometry.PointCloud()
            traj_point_cloud.points = o3d.utility.Vector3dVector([(x, y, halfh) for x, y in flat_trajectories])
            traj_point_cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in flat_trajectories])
            point_cloud += traj_point_cloud

            # Adds the map.
            planner_map_occupied: List[Tuple[float, float]] = []
            ymax, xmax = planner_map.grid.shape[:2]
            for x in range(xmax):
                for y in range(ymax):
                    if planner_map.is_occupied((x, y)):
                        xy = planner_map.to_xy((x, y))
                        planner_map_occupied.append(xy)
            map_point_cloud = o3d.geometry.PointCloud()
            map_point_cloud.points = o3d.utility.Vector3dVector([(x, y, maxh + 1) for x, y in planner_map_occupied])
            map_point_cloud.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in planner_map_occupied])
            point_cloud += map_point_cloud

            o3d.io.write_point_cloud(str(artifacts_dir / "point_cloud.ply"), point_cloud)

        logger.info("Saved artifacts to %s", artifacts_dir)


def main() -> None:
    """Entry point for the script to get trajectories for a trained model.

    Note that the goal locations need to be separated by a semicolon because
    of how argparse interprets strings.

    See `scripts/get_trajectories_demo.sh` for an example of how to use this
    script.
    """

    configure_logging(use_tqdm=True)

    parser = argparse.ArgumentParser(description="Get trajectories for a trained model.")
    parser.add_argument(
        "--planner",
        type=str,
        choices=get_args(PlannerType),
        default="gradient",
        help="The planner to use.",
    )
    parser.add_argument(
        "--xy",
        nargs=2,
        type=float,
        help="The starting location as an (X, Y) pair.",
        required=True,
    )
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
        required=False,
    )
    args = parser.parse_args()

    # Gets the device (GPU, CPU, Metal...)
    os.environ["USE_FP64"] = "1"
    device = AutoDevice.detect_device()

    # Loads the model from the model root.
    ckpt_path, config_path = Path(args.ckpt_path), Path(args.config_path)
    model, task = load_clip_sdf_model_from_ckpt_and_config(ckpt_path, config_path, device)

    # Gets the dataset and planner from the given model.
    floor_ceil_heights = float(args.floor_height), float(args.ceil_height)
    dataset = cast(Dataset[PosedRGBDItem], task.get_dataset("train"))
    planner = get_planner(model, task, cast(PlannerType, args.planner), floor_ceil_heights, device, dataset)

    # Gets the starting XY coordinates, the goal locations and the artifacts directory from arguments.
    start_xy = float(args.xy[0]), float(args.xy[1])
    goals = cast(List[str], args.goals.split(";"))
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else None

    print_trajectories(start_xy, goals, planner, floor_ceil_heights, artifacts_dir, dataset)


if __name__ == "__main__":
    main()
