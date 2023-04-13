import functools
import heapq
import math
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from usa.planners.base import Map, Planner

Heuristic = Literal["manhattan", "euclidean", "octile", "chebyshev"]


def neighbors(pt: Tuple[int, int]) -> List[Tuple[int, int]]:
    return [(pt[0] + dx, pt[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]


class AStarPlanner(Planner):
    def __init__(
        self,
        heuristic: Heuristic,
        is_occ: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
    ) -> None:
        super().__init__()

        self.heuristic = heuristic
        self.is_occ = is_occ
        self.origin = origin
        self.resolution = resolution

    def point_is_occupied(self, x: int, y: int) -> bool:
        occ_map = self.get_map()
        if x < 0 or y < 0 or x >= occ_map.grid.shape[1] or y >= occ_map.grid.shape[0]:
            return True
        return bool(occ_map.grid[y][x])

    def xy_is_occupied(self, x: float, y: float) -> bool:
        return self.point_is_occupied(*self.to_pt((x, y)))

    def to_pt(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        return self.get_map().to_pt(xy)

    def to_xy(self, pt: Tuple[int, int]) -> Tuple[float, float]:
        return self.get_map().to_xy(pt)

    def compute_heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        if self.heuristic == "manhattan":
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        if self.heuristic == "euclidean":
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        if self.heuristic == "octile":
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return (dx + dy) + (1 - 2) * min(dx, dy)
        if self.heuristic == "chebyshev":
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        raise ValueError(f"Unknown heuristic: {self.heuristic}")

    def is_in_line_of_sight(self, start_pt: Tuple[int, int], end_pt: Tuple[int, int]) -> bool:
        """Checks if there is a line-of-sight between two points.

        Implements using Bresenham's line algorithm.

        Args:
            start_pt: The starting point.
            end_pt: The ending point.

        Returns:
            Whether there is a line-of-sight between the two points.
        """

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]

        if abs(dx) > abs(dy):
            if dx < 0:
                start_pt, end_pt = end_pt, start_pt
            for x in range(start_pt[0], end_pt[0] + 1):
                yf = start_pt[1] + (x - start_pt[0]) / dx * dy
                # if self.point_is_occupied(x, int(yf)):
                #     return False
                for y in list({math.floor(yf), math.ceil(yf)}):
                    if self.point_is_occupied(x, y):
                        return False

        else:
            if dy < 0:
                start_pt, end_pt = end_pt, start_pt
            for y in range(start_pt[1], end_pt[1] + 1):
                xf = start_pt[0] + (y - start_pt[1]) / dy * dx
                # if self.point_is_occupied(int(x), y):
                #     return False
                for x in list({math.floor(xf), math.ceil(xf)}):
                    if self.point_is_occupied(x, y):
                        return False

        return True

    def clean_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Cleans up the final path.

        This implements a simple algorithm where, given some current position,
        we find the last point in the path that is in line-of-sight, and then
        we set the current position to that point. This is repeated until we
        reach the end of the path. This is not particularly efficient, but
        it's simple and works well enough.

        Args:
            path: The path to clean up.

        Returns:
            The cleaned up path.
        """

        cleaned_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if self.is_in_line_of_sight(path[i], path[j]):
                    break
            else:
                j = i + 1
            cleaned_path.append(path[j])
            i = j
        return cleaned_path

    def get_unoccupied_neighbor(
        self, pt: Tuple[int, int], goal_pt: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        if not self.point_is_occupied(*pt):
            return pt

        # This is a sort of hack to deal with points that are close to an edge.
        # If the start point is occupied, we check adjacent points until we get
        # one which is unoccupied. If we can't find one, we throw an error.
        neighbor_pts = neighbors(pt)
        if goal_pt is not None:
            goal_pt_non_null = goal_pt
            neighbor_pts.sort(key=lambda n: self.compute_heuristic(n, goal_pt_non_null))
        for neighbor_pt in neighbor_pts:
            if not self.point_is_occupied(*neighbor_pt):
                return neighbor_pt

        raise ValueError("No reachable points")

    def get_reachable_points(self, start_pt: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Gets all reachable points from a given starting point.

        Args:
            start_pt: The starting point

        Returns:
            The set of all reachable points
        """

        start_pt = self.get_unoccupied_neighbor(start_pt)

        reachable_points: Set[Tuple[int, int]] = set()
        to_visit = [start_pt]
        while to_visit:
            pt = to_visit.pop()
            if pt in reachable_points:
                continue
            reachable_points.add(pt)
            for new_pt in neighbors(pt):
                if new_pt in reachable_points:
                    continue
                if self.point_is_occupied(new_pt[0], new_pt[1]):
                    continue
                to_visit.append(new_pt)
        return reachable_points

    def plan(
        self,
        start_xy: Tuple[float, float],
        end_xy: Optional[Tuple[float, float]] = None,
        end_goal: Optional[str] = None,
        remove_line_of_sight_points: bool = True,
    ) -> List[Tuple[float, float]]:
        assert end_goal is None and end_xy is not None, "AStarPlanner doesn't implement location queries"

        start_pt, end_pt = self.to_pt(start_xy), self.to_pt(end_xy)

        # Checks that both points are unoccupied.
        start_pt = self.get_unoccupied_neighbor(start_pt, end_pt)
        end_pt = self.get_unoccupied_neighbor(end_pt, start_pt)

        # Implements A* search.
        q = [(0, start_pt)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_pt: None}
        cost_so_far: Dict[Tuple[int, int], float] = {start_pt: 0.0}
        while q:
            _, current = heapq.heappop(q)

            if current == end_pt:
                break

            for nxt in neighbors(current):
                if self.point_is_occupied(nxt[0], nxt[1]):
                    continue
                new_cost = cost_so_far[current] + self.compute_heuristic(current, nxt)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.compute_heuristic(end_pt, nxt)
                    heapq.heappush(q, (priority, nxt))  # type: ignore
                    came_from[nxt] = current

        else:
            raise ValueError("No path found")

        # Reconstructs the path.
        path = []
        current = end_pt
        while current != start_pt:
            path.append(current)
            prev = came_from[current]
            if prev is None:
                break
            current = prev
        path.append(start_pt)
        path.reverse()

        # Clean up the path.
        if remove_line_of_sight_points:
            path = self.clean_path(path)

        return [start_xy] + [self.to_xy(pt) for pt in path[1:-1]] + [end_xy]

    def is_valid_starting_point(self, xy: Tuple[float, float]) -> bool:
        return not self.point_is_occupied(*self.to_pt(xy))

    @functools.lru_cache
    def get_map(self) -> Map:
        return Map(
            grid=self.is_occ,
            resolution=self.resolution,
            origin=self.origin,
        )

    def score_locations(self, end_goal: str, xyzs: List[Tuple[float, float, float]]) -> List[float]:
        raise NotImplementedError("AStarPlanner doesn't implement location queries")
