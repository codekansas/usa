from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data.dataset import Dataset

from usa.tasks.datasets.posed_rgbd import get_bounds, get_poses, iter_xyz
from usa.tasks.datasets.types import PosedRGBDItem


@dataclass
class Map:
    grid: np.ndarray
    resolution: float
    origin: tuple[float, float]

    def to_pt(self, xy: tuple[float, float]) -> tuple[int, int]:
        return (
            int((xy[0] - self.origin[0]) / self.resolution + 0.5),
            int((xy[1] - self.origin[1]) / self.resolution + 0.5),
        )

    def to_xy(self, pt: tuple[int, int]) -> tuple[float, float]:
        return (
            pt[0] * self.resolution + self.origin[0],
            pt[1] * self.resolution + self.origin[1],
        )

    def is_occupied(self, pt: tuple[int, int]) -> bool:
        return bool(self.grid[pt[1], pt[0]])


def get_occupancy_map_from_dataset(
    ds: Dataset[PosedRGBDItem],
    cell_size: float,
    occ_height_range: tuple[float, float],
    occ_threshold: int = 100,
    clear_around_bot_radius: float = 0.0,
    cache_dir: Path | None = None,
    ignore_cached: bool = True,
) -> Map:
    """Gets the occupancy map from the given dataset.

    This employs a voting strategy to smooth out noisy points. We detect if
    there are points in some height range, and if so, we add one vote to the
    cell. We then threshold the votes to get the occupancy map.

    Args:
        ds: The dataset to get the occupancy map from.
        cell_size: The size of each cell in the occupancy map.
        occ_height_range: The height range to consider occupied.
        occ_threshold: The count threshold to consider a cell occupied.
        clear_around_bot_radius: The radius to clear around the robot.
        cache_dir: The directory to cache the occupancy map in.
        ignore_cached: Whether to ignore the cached occupancy map.
            The constructed occupancy map doesn't change for different
            models

    Returns:
        The occupancy map.
    """

    bounds = get_bounds(ds, cache_dir)
    origin = (bounds.xmin, bounds.ymin)
    resolution = cell_size

    min_height, max_height = occ_height_range
    args = (min_height, max_height, occ_threshold, clear_around_bot_radius)
    args_str = "_".join(str(a) for a in args)
    cache_loc = None if cache_dir is None else cache_dir / f"occ_map_{args_str}.npy"

    if not ignore_cached and cache_loc is not None and cache_loc.is_file():
        occ_map = np.load(cache_loc)

    else:
        xbins, ybins = int(bounds.xdiff / resolution) + 1, int(bounds.ydiff / resolution) + 1
        counts: Tensor | None = None
        any_counts: Tensor | None = None

        # Counts the number of points in each cell.
        with torch.no_grad():
            for xyz, mask_tensor in iter_xyz(ds, "Occupancy Map"):
                xyz = xyz[~mask_tensor]
                xy = xyz[:, :2]

                xs = ((xy[:, 0] - origin[0]) / resolution).floor().long()
                ys = ((xy[:, 1] - origin[1]) / resolution).floor().long()

                if counts is None:
                    counts = xy.new_zeros((ybins, xbins), dtype=torch.long).flatten()
                if any_counts is None:
                    any_counts = xy.new_zeros((ybins, xbins), dtype=torch.bool).flatten()

                # Counts the number of occupying points in each cell.
                occ_xys = (xyz[:, 2] >= min_height) & (xyz[:, 2] <= max_height)

                if len(occ_xys) != 0:
                    occ_inds = ys[occ_xys] * xbins + xs[occ_xys]
                    counts.index_add_(0, occ_inds, torch.ones_like(xs[occ_xys], dtype=torch.long))

                # Keeps track of the cells that have any points from anywhere.
                inds = ys * xbins + xs
                any_counts.index_fill_(0, inds, True)

            assert counts is not None and any_counts is not None, "No points in the dataset"
            counts = counts.reshape((ybins, xbins))
            any_counts = any_counts.reshape((ybins, xbins))

            # Clears an area around the robot's poses.
            if clear_around_bot_radius > 0:
                poses = get_poses(ds, cache_dir=cache_dir)  # (T, 4, 4) array
                pose_xys = poses[:, :2, 3]
                for x, y in pose_xys:
                    x0, x1 = x - clear_around_bot_radius, x + clear_around_bot_radius
                    y0, y1 = y - clear_around_bot_radius, y + clear_around_bot_radius
                    x0, x1 = int((x0 - origin[0]) / resolution), int((x1 - origin[0]) / resolution)
                    y0, y1 = int((y0 - origin[1]) / resolution), int((y1 - origin[1]) / resolution)
                    x0, x1 = min(max(x0, 0), xbins), min(max(x1, 0), xbins)
                    y0, y1 = min(max(y0, 0), ybins), min(max(y1, 0), ybins)
                    counts[y0:y1, x0:x1] = 0
                    any_counts[y0:y1, x0:x1] = True

            occ_map = ((counts >= occ_threshold) | ~any_counts).cpu().numpy()

            if cache_loc is not None:
                cache_loc.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_loc, occ_map)

    return Map(occ_map, resolution, origin)


class Planner(nn.Module, ABC):
    @abstractmethod
    def is_valid_starting_point(self, xy: tuple[float, float]) -> bool:
        """Checks if the starting point is valid.

        Args:
            xy: The starting point.

        Returns:
            True if the starting point is valid.
        """

    @abstractmethod
    def get_map(self) -> Map:
        """Returns the map that this planner is using.

        Returns:
            The map.
        """

    @abstractmethod
    def plan(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float] | None = None,
        end_goal: str | None = None,
    ) -> list[tuple[float, float]]:
        """Plan a path from start_xy to an ending location.

        Note that either `end_xy` or `end_goal` must be specified, but not both.

        Args:
            start_xy: The start position.
            end_xy: The end position.
            end_goal: The end goal, specified as a string.

        Returns:
            A list of waypoints from start_xy to end_xy.
        """

    @abstractmethod
    def score_locations(self, end_goal: str, xyzs: list[tuple[float, float, float]]) -> list[float]:
        """Scores the locations by their semantic similarity to the goal.

        Args:
            end_goal: The end goal, specified as a string.
            xyzs: The positions to score.

        Returns:
            The scores.
        """
