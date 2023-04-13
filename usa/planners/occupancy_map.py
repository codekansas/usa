from pathlib import Path

import cv2
import numpy as np
from torch.utils.data.dataset import Dataset

from usa.planners.base import Map, Planner, get_occupancy_map_from_dataset
from usa.planners.common import AStarPlanner as AStarPlannerBase, Heuristic
from usa.tasks.datasets.types import PosedRGBDItem


class OccupancyMapPlanner(Planner):
    def __init__(
        self,
        dataset: Dataset[PosedRGBDItem],
        resolution: float,
        floor_height: float = 0.1,
        ceil_height: float = 1.3,
        occ_avoid_radius: float = 0.5,
        clear_around_bot_radius: float = 0.3,
        cache_dir: Path | None = None,
    ) -> None:
        """Initializes the OccupancyMapPlanner.

        Args:
            dataset: The dataset to use for planning.
            resolution: The resolution of the occupancy map.
            floor_height: The height of the floor.
            ceil_height: The height of the ceiling.
            occ_avoid_radius: The radius to avoid obstacles by.
            clear_around_bot_radius: The radius to clear around the robot.
            cache_dir: The directory to save cache artifacts in.
        """

        super().__init__()

        self.floor_height = floor_height
        self.ceil_height = ceil_height
        self.occ_avoid_radius = occ_avoid_radius
        self.clear_around_bot_radius = clear_around_bot_radius

        # Gets the last occupancy map from the dataset, to use for planning.
        occ_height_range = (self.floor_height, self.ceil_height)
        occupancy_map = get_occupancy_map_from_dataset(
            dataset,
            resolution,
            occ_height_range,
            clear_around_bot_radius=self.clear_around_bot_radius,
            cache_dir=cache_dir,
        )
        self.origin = occupancy_map.origin
        self.resolution = occupancy_map.resolution

        # The array of occupied positions.
        is_occ = occupancy_map.grid.astype(np.float32)
        radius = int(np.round(self.occ_avoid_radius / self.resolution))
        is_occ = cv2.dilate(is_occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)))
        self.is_occ = is_occ

    def score_locations(self, end_goal: str, xyzs: list[tuple[float, float, float]]) -> list[float]:
        raise NotImplementedError("OccupancyMapPlanner doesn't implement location queries")


class AStarPlanner(OccupancyMapPlanner):
    def __init__(
        self,
        dataset: Dataset[PosedRGBDItem],
        heuristic: Heuristic,
        resolution: float,
        cache_dir: Path | None = None,
        floor_height: float = 0.1,
        ceil_height: float = 1.3,
    ) -> None:
        """Initializes the AStarPlanner.

        Args:
            dataset: The dataset to use for planning.
            heuristic: The heuristic to use for planning.
            resolution: The resolution of the occupancy map.
            cache_dir: The directory to save cache artifacts to.
            floor_height: The height of the floor.
            ceil_height: The height of the ceiling.
        """

        # Initializes the OccupancyMapPlanner.
        super().__init__(
            dataset,
            resolution,
            cache_dir=cache_dir,
            floor_height=floor_height,
            ceil_height=ceil_height,
        )

        # Initializes the AStarPlanner using the occupancy map from the
        # OccupancyMapPlanner.
        self.a_star_planner = AStarPlannerBase(
            heuristic=heuristic,
            is_occ=self.is_occ,
            origin=self.origin,
            resolution=resolution,
        )

    def plan(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float] | None = None,
        end_goal: str | None = None,
    ) -> list[tuple[float, float]]:
        return self.a_star_planner.plan(start_xy, end_xy, end_goal)

    def is_valid_starting_point(self, xy: tuple[float, float]) -> bool:
        return self.a_star_planner.is_valid_starting_point(xy)

    def get_map(self) -> Map:
        return self.a_star_planner.get_map()
