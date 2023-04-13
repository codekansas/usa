import argparse
import bdb
import csv
import logging
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import ml.api as ml
import numpy as np
import pandas as pd
import seaborn as sns

from usa.evaluation.utils import EvalSet, get_planners, plot_map, plot_paths
from usa.planners.base import Map

logger = logging.getLogger(__name__)

PathStartEndCoords = list[tuple[tuple[float, float], tuple[float, float]]]


def not_in_line_of_sight(planner_map: Map, xy: np.ndarray, start_xy: tuple[float, float]) -> np.ndarray:
    """Gets a mask of the points which aren't in line-of-sight from the start.

    Args:
        planner_map: The occupancy map
        xy: The XZ coordinates of the points
        start_xy: The XZ coordinates of the start point

    Returns:
        A mask of the points which aren't in line-of-sight from the start.
    """

    def is_line_of_sight(start_xy: tuple[float, float], end_xy: tuple[float, float]) -> bool:
        start_pt = planner_map.to_pt(start_xy)
        end_pt = planner_map.to_pt(end_xy)

        if start_pt == end_pt:
            return True

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]

        if abs(dx) > abs(dy):
            if dx < 0:
                start_pt, end_pt = end_pt, start_pt
            for x in range(start_pt[0], end_pt[0] + 1):
                yf = start_pt[1] + (x - start_pt[0]) / dx * dy
                # if planner_map.grid[int(yf)][x]:
                #     return False
                for y in list({math.floor(yf), math.ceil(yf)}):
                    if planner_map.grid[y][x]:
                        return False

        else:
            if dy < 0:
                start_pt, end_pt = end_pt, start_pt
            for y in range(start_pt[1], end_pt[1] + 1):
                xf = start_pt[0] + (y - start_pt[1]) / dy * dx
                # if planner_map.grid[y][int(xf)]:
                #     return False
                for x in list({math.floor(xf), math.ceil(xf)}):
                    if planner_map.grid[y][x]:
                        return False

        return True

    vals = [is_line_of_sight(start_xy, end_xy) for end_xy in xy]

    return ~np.array(vals)


def get_eval_set(
    name: str,
    save_dir: Path,
    num_paths: int = 100,
    min_path_euclid_mul: float = 0.1,
    filter_line_of_sight_points: bool = True,
) -> tuple[EvalSet, PathStartEndCoords]:
    """Gets the evaluation set for a clip recorded with the Stretch robot.

    Args:
        name: The name of the evaluation set.
        save_dir: The directory to save the evaluation set to.
        num_paths: The number of paths to evaluate
        min_path_euclid_mul: The minimum euclidean distance between the start
            and end points of the path (multiplied by the map size)
        filter_line_of_sight_points: Whether to filter out points which are
            in line-of-sight from the start point.

    Returns:
        The evaluation set for evaluating the path length and the coordinates
        of the paths being evaluated.

    Raises:
        ValueError: If there is some validation error
    """

    planners, dataset, poses = get_planners(name, True)

    # Plots maps for each planner.
    maps_dir = save_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for planner_name, planner in planners.items():
        plot_map(planner.get_map(), maps_dir / f"{planner_name}.png")
        logger.info("Plotted map for %s", planner_name)

    # Gets the XZ coordinates (remember, using the camera frame, so Z is
    # forward while X is right).
    xs, ys = poses[:, 0, 3], poses[:, 1, 3]
    xy = np.stack([xs, ys], axis=1)

    # Gets the occupancy maps.
    planner_maps = [planner.get_map() for planner in planners.values()]

    # Removes invalid starting points.
    valid_starting_points = np.ones_like(xs, dtype=bool)
    for i, xyi in enumerate(xy):
        if any(planner_map.is_occupied(planner_map.to_pt(xyi)) for planner_map in planner_maps):
            valid_starting_points[i] = False
    xy = xy[valid_starting_points]

    # Does the first check.
    for xyi in xy:
        for planner_map in planner_maps:
            if planner_map.is_occupied(planner_map.to_pt(xyi)):
                raise ValueError("Invalid starting point - this shouldn't happen")

    # Choose some random starting points, and some random ending points which
    # are at least 1.0 meters away from the starting point.
    start_idxs = np.random.choice(len(xy), size=min(num_paths, len(xy)), replace=False)
    start_xy = xy[start_idxs]
    end_xy = np.zeros_like(start_xy)
    invalid_xy = np.zeros_like(start_xy[:, 0], dtype=bool)
    min_path_euclid_dist = min_path_euclid_mul * np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))
    for i, (start_x, start_y) in enumerate(start_xy):
        mask = np.linalg.norm(xy - np.array([start_x, start_y]), axis=1) > min_path_euclid_dist
        valid_end_xy = xy[mask]
        if filter_line_of_sight_points:
            los_arrs = [
                not_in_line_of_sight(planner_map, valid_end_xy, (start_x, start_y)) for planner_map in planner_maps
            ]
            valid_end_xy = valid_end_xy[np.array(los_arrs).any(axis=0)]
        if len(valid_end_xy) == 0:
            invalid_xy[i] = True
        else:
            end_xy[i] = valid_end_xy[np.random.choice(len(valid_end_xy))]

    # Removes invalid paths.
    start_xy = start_xy[~invalid_xy]
    end_xy = end_xy[~invalid_xy]

    # Raises an error if there are no valid paths.
    if len(start_xy) == 0:
        raise ValueError("No valid paths found")

    # Converts to expected format.
    path_coords = [((sx, sy), (ex, ey)) for (sx, sy), (ex, ey) in zip(start_xy, end_xy)]

    # Checks all the path coordinates.
    for sxy, exy in path_coords:
        for planner_map in planner_maps:
            for pt in (planner_map.to_pt(xy) for xy in (sxy, exy)):
                if planner_map.is_occupied(pt):
                    raise ValueError("This shouldn't happen")

    eval_set = EvalSet(
        name=name,
        dataset=dataset,
        planners=planners,
    )

    return eval_set, path_coords


def evaluate(eval_set: EvalSet, path_coords: PathStartEndCoords, eval_set_dir: Path) -> None:
    ml.configure_logging(use_tqdm=True)

    # Setting this for the optimizer.
    os.environ["USE_FP64"] = "1"

    logger.info("Evaluating %s on %d paths", eval_set.name, len(path_coords))

    # Creates the directory to save the evaluation set to.
    eval_dir = eval_set_dir / "path_lengths"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for planner_name, planner in sorted(eval_set.planners.items()):
        logger.info("Evaluating %s", planner_name)

        # Writes the path coordinates to a CSV file.
        path_dir = eval_dir / planner_name
        path_dir.mkdir(parents=True, exist_ok=True)

        with open(path_dir / "lengths.csv", "w", encoding="utf-8") as f:
            lengths_writer = csv.writer(f, delimiter=",")
            lengths_writer.writerow(["start_x", "start_y", "end_x", "end_y", "path_length"])

            planner_map = planner.get_map()
            paths: list[list[tuple[float, float]]] = []

            for i, (start, end) in enumerate(path_coords):
                try:
                    path = planner.plan(start, end)

                    # Checks that path start and end are correct.
                    assert np.allclose(path[0], start, atol=1e-3), f"Path start is incorrect: {path[0]} != {start}"
                    assert np.allclose(path[-1], end, atol=1e-3), f"Path end is incorrect: {path[-1]} != {end}"

                    path_length = sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
                    if i < 10:  # Don't plot too many paths.
                        plot_paths(planner_map, [path], path_dir / f"{i}.png")
                    paths.append(path)
                    lengths_writer.writerow([start[0], start[1], end[0], end[1], path_length])

                except bdb.BdbQuit:
                    raise

                except Exception:
                    logger.exception("Exception while getting path")
                    lengths_writer.writerow([start[0], start[1], end[0], end[1], None])

            # Plots all of the paths on top of the map and saves it as a PNG.
            plot_paths(planner_map, paths, eval_dir / f"{planner_name}.png")

    # Load all CSVs into a Pandas DataFrame. Uses the row index as a new
    # column called "path_id".
    dfs = []
    for planner_name, planner in sorted(eval_set.planners.items()):
        path_dir = eval_dir / planner_name
        df = pd.read_csv(path_dir / "lengths.csv")
        df["planner"] = planner_name
        df["path_id"] = df.index
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Computes the mean increase in path length relative to the best planner.
    best_planner = df.groupby("planner").mean().sort_values("path_length").index[0]
    best_planner_paths = df[df["planner"] == best_planner]

    # Adds "rel_path_length" column.
    df = df.merge(best_planner_paths[["path_id", "path_length"]], on="path_id", suffixes=("", "_best"))
    df["rel_path_length"] = df["path_length"] / df["path_length_best"]

    # Gets the mean relative path length for each planner.
    mean_rel_path_length = df.groupby("planner").mean()["rel_path_length"]
    print(mean_rel_path_length.round(3))

    # Graph the path lengths for each planner for each path. Use sns.barplot.

    # Square figure.
    # plt.figure(figsize=(20, 20))

    # Wide figure.
    plt.figure(figsize=(60, 20))

    fig, ax = plt.subplots()
    sns.barplot(
        data=df,
        x="path_id",
        y="path_length",
        hue="planner",
        ax=ax,
    )
    ax.set_title(f"Path Lengths for {eval_set.name}")
    ax.set_xlabel("Path ID")
    ax.set_ylabel("Path Length (meters)")
    fig.savefig(eval_dir / "path_lengths.png")

    logger.info("Finished evaluating %s. Wrote logs to %s", eval_set.name, eval_dir)


def main() -> None:
    """Main entry point for `path_length` evaluation CLI.

    Usage:
        python -m ml.evaluation.evaluations.path_length
    """

    ml.configure_logging(use_tqdm=True)

    parser = argparse.ArgumentParser(description="Runs path length evaluation.")
    parser.add_argument("key", help="The evaluation key to use")
    parser.add_argument("save_dir", help="Where to save results")
    parser.add_argument("-n", "--num-paths", type=int, default=100, help="Maximum number of paths to evaluate")
    parser.add_argument("-e", "--min-euclid-mul", type=float, default=0.1, help="Minimum path distance multipler")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Runs the evaluation.
    eval_set, path_coords = get_eval_set(args.key, save_dir, args.num_paths, args.min_euclid_mul)
    evaluate(eval_set, path_coords, save_dir)


if __name__ == "__main__":
    main()
