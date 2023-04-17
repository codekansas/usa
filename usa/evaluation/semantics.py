import argparse
import bdb
import csv
import logging
import os
from pathlib import Path
from typing import Iterator

import ml.api as ml
import numpy as np
import open3d as o3d
import pandas as pd
import torch

from usa.evaluation.utils import EvalSet, get_planners, plot_paths
from usa.planners.clip_sdf import ClipSdfPlanner
from usa.tasks.datasets.utils import (
    make_point_cloud_from_dataset,
    make_video_from_dataset,
)

logger = logging.getLogger(__name__)

PathStartCoordsAndGoal = list[tuple[tuple[float, float], str]]


def get_eval_set(
    name: str,
    queries: list[str],
    rotate: bool = False,
    concat_horizontal: bool = False,
) -> tuple[EvalSet, PathStartCoordsAndGoal]:
    """Gets the evaluation set for a clip recorded with the Stretch robot.

    Args:
        name: The name of the evaluation set.
        queries: The queries to use for the.
        rotate: If the video should be rotated.
        concat_horizontal: If the video should be concatenated horizontally.

    Returns:
        The evaluation set for evaluating the path length, along with the
        start coordinates and goal.
    """

    # Seed everything for reproducibility.
    ml.set_random_seed(1337)

    planners, dataset, poses = get_planners(name, False)

    # Gets the XZ coordinates (remember, using the camera frame, so Z is
    # forward while X is right).
    xs, ys = poses[:, 0, 3], poses[:, 1, 3]
    xy = np.stack([xs, ys], axis=1)

    # Removes invalid starting points.
    valid_starting_points = np.ones_like(xs, dtype=bool)
    for i, (x, y) in enumerate(xy):
        if not all(planner.is_valid_starting_point((x, y)) for planner in planners.values()):
            valid_starting_points[i] = False
    xy = xy[valid_starting_points]

    # Choose some random starting points, and some random ending points which
    # are at least 1.0 meters away from the starting point.
    start_idxs = np.random.choice(len(xy), size=len(queries), replace=False)
    start_xy = xy[start_idxs]

    # Converts to expected format.
    start_xy_list = [((sx, sy), query) for (sx, sy), query in zip(start_xy, queries)]

    eval_set = EvalSet(
        name=name,
        dataset=dataset,
        planners=planners,
        rotate=rotate,
        concat_horizontal=concat_horizontal,
    )

    return eval_set, start_xy_list


def evaluate(eval_set: EvalSet, path_coords: PathStartCoordsAndGoal, eval_set_dir: Path) -> None:
    configure_logging(use_tqdm=True)

    # Setting this for the optimizer.
    os.environ["USE_FP64"] = "1"

    logger.info("Evaluating %s on %d paths", eval_set.name, len(path_coords))

    # Makes a new directory for running the evaluation.
    eval_dir = eval_set_dir / "semantics"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Gets a point cloud for the eval set. This can be somewhat slow so only
    # get it if it doesn't already exist.
    point_cloud_path = eval_set_dir / "point_cloud.ply"
    if point_cloud_path.exists():
        point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    else:
        point_cloud = make_point_cloud_from_dataset(
            eval_set.dataset,
            batch_size=4,
            num_workers=0,
            voxel_size=5e-2,
            max_batch_points=100_000,
            max_height=1.3,  # Remove the ceiling.
        )
        o3d.io.write_point_cloud(str(point_cloud_path), point_cloud)
    logger.info("Point cloud path: %s", point_cloud_path)

    # Saves the video.
    video_path = eval_set_dir / "video.mp4"
    if not video_path.exists():
        make_video_from_dataset(
            eval_set.dataset,
            video_path,
            rotate=eval_set.rotate,
            concat_horizontal=eval_set.concat_horizontal,
        )
    logger.info("Video path: %s", video_path)

    # Saves the path point cloud.
    path_point_cloud_path = eval_set_dir / "path_point_cloud.ply"
    if not path_point_cloud_path.exists():
        poses = torch.stack([i.pose for i in eval_set.dataset])
        xyz = poses[..., :3, 3].cpu().numpy()
        path_point_cloud = o3d.geometry.PointCloud()
        path_point_cloud.points = o3d.utility.Vector3dVector(xyz)

        # Gets color gradient from blue to purple.
        path_colors = np.zeros_like(xyz)
        path_colors[:, 2] = 1
        path_colors[:, 0] = np.linspace(0, 1, len(xyz))
        path_point_cloud.colors = o3d.utility.Vector3dVector(path_colors)

        o3d.io.write_point_cloud(str(path_point_cloud_path), path_point_cloud)
    logger.info("Path point cloud path: %s", path_point_cloud_path)

    def clean_goal(goal: str) -> str:
        # Converts a generic goal string into a string to use as a file name.
        return goal.replace(" ", "_").replace("/", "_").lower()

    def interpolate_path(path: list[tuple[float, float]], resolution: float) -> Iterator[tuple[float, float]]:
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            num_points = int(np.linalg.norm(np.array(start) - np.array(end)) / resolution) + 1
            for t in np.linspace(0, 1, num_points):
                yield (1 - t) * start[0] + t * end[0], (1 - t) * start[1] + t * end[1]

    with open(eval_dir / "final_location_scores.csv", "w", encoding="utf-8") as g:
        final_location_score_writer = csv.writer(g)
        final_location_score_writer.writerow(["planner", "goal", "score"])

        for planner_name, planner in sorted(eval_set.planners.items()):
            logger.info("Evaluating %s", planner_name)

            assert isinstance(planner, ClipSdfPlanner), "Unexpected planner type"

            # Writes the path coordinates and goal to a CSV file.
            path_dir = eval_dir / planner_name
            path_dir.mkdir(parents=True, exist_ok=True)

            with open(path_dir / "lengths.csv", "w", encoding="utf-8") as f:
                lengths_writer = csv.writer(f, delimiter=",")
                lengths_writer.writerow(["start_x", "start_y", "goal", "path_length"])

                planner_map = planner.get_map()
                paths: list[list[tuple[float, float]]] = []

                for start, goal in path_coords:
                    try:
                        path = planner.plan(start, end_goal=goal)

                        # Checks that path start and end are correct.
                        assert np.allclose(path[0], start, atol=1e-3), f"Path start is incorrect: {path[0]} != {start}"

                        path_length = sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
                        plot_paths(planner_map, [path], path_dir / f"{clean_goal(goal)}.png")
                        paths.append(path)
                        lengths_writer.writerow([start[0], start[1], goal, path_length])

                        # Gets the scores for all the points in the point cloud.
                        point_cloud_xyz = [(p[0], p[1], p[2]) for p in point_cloud.points]
                        x, y = path[-1]
                        point_cloud_xyz += [(x, y, planner.min_z), (x, y, planner.max_z)]
                        scores = planner.score_locations(goal, point_cloud_xyz)
                        point_cloud_scores, final_location_score = scores[:-2], max(scores[-2], scores[-1])

                        # Gets the points higher than some threshold.
                        top_pts = np.argsort(point_cloud_scores)
                        top_pts_scores = np.array(point_cloud_scores)[top_pts]
                        min_top_score = (top_pts_scores[-1] - top_pts_scores[0]) * 0.9 + top_pts_scores[0]
                        top_pts = top_pts[top_pts_scores > min_top_score]
                        top_pts_xyz = np.array(point_cloud_xyz)[top_pts]
                        top_point_cloud_path = path_dir / f"{clean_goal(goal)}_top.ply"
                        top_point_cloud = o3d.geometry.PointCloud()
                        top_point_cloud.points = o3d.utility.Vector3dVector(top_pts_xyz)
                        top_point_cloud.paint_uniform_color([1, 0, 0])
                        o3d.io.write_point_cloud(str(top_point_cloud_path), top_point_cloud)

                        # Writes the score for the final location.
                        final_location_score_writer.writerow([planner_name, goal, final_location_score])

                        # Saves the point cloud with the path overlaid.
                        point_cloud_path = path_dir / f"{clean_goal(goal)}_path.ply"
                        path_points = o3d.geometry.PointCloud()
                        path_xy, path_z = np.array(list(interpolate_path(path, 0.05))), 0.5
                        path_xyz = np.concatenate([path_xy, np.full_like(path_xy[:, :1], path_z)], axis=1)

                        # Gets fully-red path.
                        # path_colors = np.ones_like(path_xyz)
                        # path_colors[:, 1:] = 0

                        # Gets color gradient from red to green.
                        path_colors = np.zeros_like(path_xyz)
                        path_colors[:, 0] = np.linspace(1, 0, len(path_xyz))
                        path_colors[:, 1] = np.linspace(0, 1, len(path_xyz))

                        path_points.points = o3d.utility.Vector3dVector(path_xyz)
                        path_points.colors = o3d.utility.Vector3dVector(path_colors)
                        o3d.io.write_point_cloud(str(point_cloud_path), path_points)

                    except bdb.BdbQuit:
                        raise

                    except Exception:
                        logger.exception("Exception while getting path")
                        lengths_writer.writerow([start[0], start[1], goal, None])

                # Plots all of the paths on top of the map and saves it as a PNG.
                plot_paths(planner_map, paths, eval_dir / f"{planner_name}.png")

    # Aggregates relative scores into a metric.
    df = pd.read_csv(eval_dir / "final_location_scores.csv")
    df = df.merge(df.groupby("goal").max()["score"], on="goal", suffixes=("", "_max"))
    df["rel_score"] = df["score"] / df["score_max"]
    mean_rel_score = df.groupby("planner")["rel_score"].mean()
    print(mean_rel_score.round(3))


def main() -> None:
    """Main entry point for the `semantics` evaluation CLI.

    Usage:
        python -m ml.evaluation.evaluations.semantics

    Raises:
        KeyError: If the evaluation set is not found.
    """

    ml.configure_logging(use_tqdm=True)

    parser = argparse.ArgumentParser(description="Runs semantic evaluation.")
    parser.add_argument("key", help="The evaluation key to use")
    parser.add_argument("save_dir", help="Where to save results")
    parser.add_argument("-n", "--num-paths", type=int, default=100, help="Maximum number of paths to evaluate")
    parser.add_argument("-e", "--min-euclid-mul", type=float, default=0.1, help="Minimum path distance multipler")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    eval_set_to_queries = {
        "lab_stretch": [
            "Computer desk chair",
            "A wooden box",
            "A man sitting at a computer",
            "Desktop computer",
            "Doorway",
            "Shelves",
        ],
        "kitchen_stretch": [
            "The coffee machine",
            "A refrigerator full of beverages",
            "Some snacks",
            "The row of stools",
            "The man sitting at a laptop",
        ],
        "chess_stretch": [
            "A chess board",
            "A comfortable chair",
            "A conference room",
        ],
        "lab_r3d": [
            "Computer desk chair",
            "A wooden box",
            "A computer desk",
            "Desktop computer",
            "Doorway",
            "Shelves",
        ],
        "studio_r3d": [
            "The mitre saw",
            "The drill press",
            "A cabinet with green organization boxes",
        ],
        "replica_apt_3_mnp": [
            "The bicycle",
            "Jackets hanging in the closet",
            "A kitchen sink",
            "A video game kitchen sink",
        ],
        "chris_lab": [
            "A computer desk",
            "A desk chair",
        ],
    }

    if args.key not in eval_set_to_queries:
        raise KeyError(f"{args.key} is not a valid eval set")
    queries = eval_set_to_queries[args.key]

    rotate_image = {
        "lab_stretch",
        "kitchen_stretch",
        "chess_stretch",
    }
    concat_horizontal = {
        "lab_stretch",
        "kitchen_stretch",
        "chess_stretch",
        "lab_r3d",
        "studio_r3d",
    }

    # Runs the evaluation.
    eval_set, path_coords = get_eval_set(
        args.key,
        queries,
        rotate=args.key in rotate_image,
        concat_horizontal=args.key in concat_horizontal,
    )
    evaluate(eval_set, path_coords, save_dir)


if __name__ == "__main__":
    main()
