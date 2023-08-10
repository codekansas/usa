import functools
import logging
from pathlib import Path
from typing import Type, cast

import ml.api as ml
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data.dataset import Dataset

from usa.models.point2emb import Point2EmbModel
from usa.planners.base import Map, Planner, get_occupancy_map_from_dataset
from usa.planners.common import AStarPlanner as AStarPlannerBase, Heuristic
from usa.tasks.clip_sdf import ClipSdfTask
from usa.tasks.datasets.types import PosedRGBDItem

logger = logging.getLogger(__name__)


class ClipSdfPlanner(Planner):
    def __init__(
        self,
        dataset: Dataset[PosedRGBDItem],
        model: Point2EmbModel,
        task: ClipSdfTask,
        device: Type[ml.BaseDevice],
        resolution: float,
        occ_avoid_radius: float = 0.3,
        floor_height: float = 0.1,
        ceil_height: float = 1.3,
        cache_dir: Path | None = None,
    ) -> None:
        """Initializes the ClipSdfPlanner.

        Args:
            dataset: The dataset to use for planning.
            model: The model to use for planning.
            task: The task for training the model.
            device: The device for running the model (just use AutoDevice).
            resolution: The resolution of the occupancy grid. This is only
                used for generating the map grid, and for grid-based planners.
            occ_avoid_radius: The radius to avoid obstacles by.
            floor_height: The maximum height of a point to consider it part
                of the floor (i.e., drivable space).
            ceil_height: The minimum height above the camera to consider a
                point part of the ceiling (i.e., not drivable space).
            cache_dir: The directory to cache the occupancy map in.
        """

        super().__init__()

        self.model = model
        self.task = task
        self.dataset = dataset
        self.device = device
        self.occ_avoid_radius = occ_avoid_radius
        self.floor_height = floor_height
        self.ceil_height = ceil_height
        self.min_z = self.floor_height + self.occ_avoid_radius
        self.max_z = self.ceil_height - self.occ_avoid_radius
        self.occ_map = get_occupancy_map_from_dataset(
            self.dataset,
            resolution,
            (self.min_z, self.max_z),
            cache_dir=cache_dir,
        )

        # Don't train the model.
        self.model.eval().requires_grad_(False)

    def get_clip_and_sdf(self, xyz: list[tuple[float, float, float]]) -> tuple[Tensor, list[float]]:
        xyz_tensor = torch.tensor(xyz)
        xyz_tensor = self.device.tensor_to(xyz_tensor)
        preds = self.model(xyz_tensor)
        clip_emb_size = preds.size(-1) - 1
        clip_preds, sdf_preds = torch.split(preds, [clip_emb_size, 1], dim=-1)
        return clip_preds.cpu(), [s.item() for s in sdf_preds.cpu()]

    def get_grid_xyzs(self) -> tuple[Tensor, Tensor, Tensor]:
        y_pixels, x_pixels = self.occ_map.grid.shape

        def occ_map_pt_to_xy(pt: tuple[int, int]) -> tuple[float, float]:
            return (
                pt[0] * self.occ_map.resolution + self.occ_map.origin[0],
                pt[1] * self.occ_map.resolution + self.occ_map.origin[1],
            )

        x_min, y_min = occ_map_pt_to_xy((0, 0))
        x_max, y_max = occ_map_pt_to_xy((x_pixels, y_pixels))

        # Gets the X, Y, and Z centers for each element in the grid.
        xs, ys, zs = torch.meshgrid(
            torch.arange(x_min, x_max, self.occ_map.resolution),
            torch.arange(y_min, y_max, self.occ_map.resolution),
            #torch.tensor([self.min_z, self.max_z]),
            #torch.tensor([self.min_z, (self.min_z + self.max_z) / 2,  self.max_z]),
            torch.arange(self.min_z, self.max_z, 0.2),
            indexing="xy",
        )

        return xs, ys, zs

    @functools.lru_cache
    def get_map(self) -> Map:
        origin = self.occ_map.origin
        xs, ys, zs = self.get_grid_xyzs()

        # Combines to a single vector.
        xyzs = torch.stack([xs, ys, zs], dim=-1)
        xyzs = xyzs.reshape(-1, 3)
        xyzs = self.device.tensor_to(xyzs)

        # Need to query in chunks to avoid memory issues.
        all_sdfs = []
        for xyz_chunk in tqdm.tqdm(torch.split(xyzs, 64)):
            # This is the old implementation, which just checks if the center
            # of a cell is occupied when building the occupancy grid.
            # sdf = self.model(xyz_chunk)[:, -1]
            # all_sdfs.append((sdf < self.occ_avoid_radius).cpu())

            # Thresholds the SDFs for the corners. This is kind of inefficient
            # because there is some duplication between the corners, but it
            # lets us be more flexible with how we specify the points within
            # the cell to sample when checking for occupancy.
            res = self.occ_map.resolution
            chunk_xs, chunk_ys, chunk_zs = torch.unbind(xyz_chunk, dim=-1)
            chunk_sdfs = []
            for dx, dy in [(0.5, 0.5), (0, 0), (0, 1), (1, 0), (1, 1)]:
                xyz_corner_chunk = torch.stack(
                    [
                        chunk_xs - (res / 2) + (dx * res),
                        chunk_ys - (res / 2) + (dy * res),
                        chunk_zs,
                    ],
                    dim=-1,
                )
                sdf = self.model(xyz_corner_chunk)[:, -1]
                chunk_sdfs.append((sdf < self.occ_avoid_radius).cpu())
            all_sdfs.append(torch.stack(chunk_sdfs, dim=-1).any(dim=-1))

        is_occ = torch.cat(all_sdfs, dim=0).view(xs.shape).any(dim=-1, keepdim=True).numpy()

        return Map(
            grid=is_occ,
            origin=origin,
            resolution=self.occ_map.resolution,
        )

    def score_locations(self, end_goal: str, xyzs: list[tuple[float, float, float]]) -> list[float]:
        with torch.no_grad():
            tokens = self.device.tensor_to(self.task.clip.tokenizer.tokenize(end_goal))
            goal_embs = self.task.clip.linguistic(tokens)
            xyzs_tensor = torch.tensor(xyzs)
            scores: list[float] = []
            for xyz_chunk in xyzs_tensor.split(256):
                xyz_chunk = self.device.tensor_to(xyz_chunk)
                clip_preds = self.model(xyz_chunk)[:, :-1]
                clip_scores = goal_embs @ clip_preds.T
                scores.extend(clip_scores.squeeze(0).cpu().tolist())
            return scores


class AStarPlanner(ClipSdfPlanner):
    def __init__(
        self,
        dataset: Dataset[PosedRGBDItem],
        model: Point2EmbModel,
        task: ClipSdfTask,
        device: Type[ml.BaseDevice],
        heuristic: Heuristic,
        resolution: float,
        cache_dir: Path | None = None,
        floor_height: float = 0.1,
        ceil_height: float = 1.3,
        occ_avoid_radius: float = 0.3
    ) -> None:
        super().__init__(
            dataset=dataset,
            model=model,
            task=task,
            device=device,
            resolution=resolution,
            cache_dir=cache_dir,
            floor_height=floor_height,
            ceil_height=ceil_height,
            occ_avoid_radius = occ_avoid_radius
        )

        # Gets the map from the parent class.
        clip_sdf_map = super().get_map()

        # Initializes the AStarPlanner using the occupancy map.
        self.a_star_planner = AStarPlannerBase(
            heuristic=heuristic,
            is_occ=clip_sdf_map.grid,
            origin=clip_sdf_map.origin,
            resolution=clip_sdf_map.resolution
        )

    def plan(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float] | None = None,
        end_goal: str | None = None,
        remove_line_of_sight_points: bool = True,
    ) -> list[tuple[float, float]]:
        if end_goal is not None:
            assert end_xy is None, "Cannot specify both end_xy and end_goal"
            end_xy = self.get_end_xy(start_xy, end_goal)
        
        return self.a_star_planner.plan(
            start_xy=start_xy,
            end_xy=end_xy,
            remove_line_of_sight_points=remove_line_of_sight_points,
        )

    @functools.lru_cache
    def get_score_map(self, end_goal: str) -> np.ndarray:
        tokens = self.device.tensor_to(self.task.clip.tokenizer.tokenize(end_goal))
        embs = self.task.clip.linguistic(tokens)

        xs, ys, zs = self.get_grid_xyzs()

        # Combines to a single vector.
        xyzs = torch.stack([xs, ys, zs], dim=-1)
        xyzs = xyzs.reshape(-1, 3)
        xyzs = self.device.tensor_to(xyzs)

        # Need to query in chunks to avoid memory issues.
        all_clip_sims = []
        for xyz_chunk in tqdm.tqdm(torch.split(xyzs, 64)):
            clip_embs = self.model(xyz_chunk)[:, :-1]
            #tok_scores = embs @ clip_embs.T
            tok_scores = torch.cosine_similarity(embs, clip_embs)
            all_clip_sims.append(tok_scores.flatten(0).detach().cpu())

        return torch.cat(all_clip_sims, dim=0).view(xs.shape).max(dim=-1, keepdim=True).values.numpy()

    def get_end_xy(self, start_xy: tuple[float, float], end_goal: str) -> tuple[float, float]:
        score_map = self.get_score_map(end_goal)
        start_pt = self.a_star_planner.to_pt(start_xy)
        reachable_pts = self.a_star_planner.get_reachable_points(start_pt)
        xs, ys = zip(*reachable_pts)
        reachable_map = np.zeros_like(score_map, dtype=bool)
        reachable_map[ys, xs] = True

        # Gets the (X, Y) index of the highest scoring unoccupied point.
        unocc_points = np.argwhere(reachable_map)
        best_point_index = np.argmax(score_map[unocc_points[:, 0], unocc_points[:, 1]])
        y, x, _ = unocc_points[best_point_index]

        assert not self.a_star_planner.point_is_occupied(x, y), "Best point is occupied"

        # Converts to (X, Y) coordinates.
        return self.a_star_planner.to_xy((x, y))

    def is_valid_starting_point(self, xy: tuple[float, float]) -> bool:
        return self.a_star_planner.is_valid_starting_point(xy)

    @functools.lru_cache
    def get_map(self) -> Map:
        return self.a_star_planner.get_map()


class GradientPlanner(ClipSdfPlanner):
    def __init__(
        self,
        dataset: Dataset[PosedRGBDItem],
        model: Point2EmbModel,
        task: ClipSdfTask,
        device: Type[ml.BaseDevice],
        lr: float = 1e-2,
        dist_loss_weight: float = 1.0,
        spacing_loss_weight: float = 1.0,
        occ_loss_weight: float = 25.0,
        sim_loss_weight: float = 15.0,
        num_optimization_steps: int = 1000,
        min_distance: float = 1e-5,
        cache_dir: Path | None = None,
        floor_height: float = 0.1,
        ceil_height: float = 1.3,
        occ_avoid_radius: float = 0.3
    ) -> None:
        # Constant resolution.
        visualization_resolution = 0.025
        planner_resolution = 0.1

        super().__init__(
            dataset=dataset,
            model=model,
            task=task,
            device=device,
            resolution=visualization_resolution,
            cache_dir=None if cache_dir is None else cache_dir / "gradient",
            floor_height=floor_height,
            ceil_height=ceil_height,
            occ_avoid_radius = occ_avoid_radius
        )

        # Gradient planner parameters.
        self.lr = lr
        self.dist_loss_weight = dist_loss_weight
        self.spacing_loss_weight = spacing_loss_weight
        self.occ_loss_weight = occ_loss_weight
        self.sim_loss_weight = sim_loss_weight
        self.num_optimization_steps = num_optimization_steps
        self.min_distance = min_distance

        # Base planner, which is used to generate a "seed" path.
        self.base_planner = AStarPlanner(
            dataset=dataset,
            model=model,
            task=task,
            device=device,
            resolution=planner_resolution,
            heuristic="euclidean",
            cache_dir=None if cache_dir is None else cache_dir / "base_planner",
            floor_height=floor_height,
            ceil_height=ceil_height,
            occ_avoid_radius = occ_avoid_radius
        )

    def get_target_query_emb(self, end_goal: str) -> Tensor:
        tokens = self.device.tensor_to(self.task.clip.tokenizer.tokenize(end_goal))
        embs = self.task.clip.linguistic(tokens)
        return embs

    def plan(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float] | None = None,
        end_goal: str | None = None,
    ) -> list[tuple[float, float]]:
        assert end_xy is not None or end_goal is not None, "Must specify either end_xy or end_goal"
        assert end_xy is None or end_goal is None, "Must specify either end_xy or end_goal, not both"

        # Gets the end goal embedding.
        target_emb = None if end_goal is None else self.get_target_query_emb(end_goal)

        # Gets a seed path from the base planner.
        seed_path = self.base_planner.plan(start_xy, end_xy, end_goal, remove_line_of_sight_points=False)
        #print(seed_path)
        xys = self.device.tensor_to(torch.tensor(seed_path))
        xys.requires_grad_(True)

        def get_losses(xys: Tensor) -> dict[str, Tensor]:
            # Loss for avoiding obstacles.
            waypoint_xyz_min = torch.cat([xys[1:], torch.full_like(xys[1:, :1], self.min_z)], dim=-1)
            waypoint_xyz_mid = torch.cat([xys[1:], torch.full_like(xys[1:, :1], (self.max_z + self.min_z)/2)], dim=-1)
            waypoint_xyz_max = torch.cat([xys[1:], torch.full_like(xys[1:, :1], self.max_z)], dim=-1)
            preds_min = self.model(waypoint_xyz_min)
            preds_mid = self.model(waypoint_xyz_mid)
            preds_max = self.model(waypoint_xyz_max)
            clip_preds_min, sdf_preds_min = preds_min[-1:, :-1], preds_min[..., -1]
            clip_preds_mid, sdf_preds_mid = preds_mid[-1:, :-1], preds_mid[..., -1]
            clip_preds_max, sdf_preds_max = preds_max[-1:, :-1], preds_max[..., -1]
            sdf_preds = torch.stack([sdf_preds_min, sdf_preds_max], dim=-1).min(-1).values
            #sdf_preds = torch.stack([sdf_preds_min, sdf_preds_mid, sdf_preds_max], dim=-1).min(-1).values
            # is_inside_obs = sdf_preds < self.occ_avoid_radius + 1e-3
            occ_loss = torch.clamp_min(self.occ_avoid_radius - sdf_preds, 0).sum()

            norms = torch.norm(xys[1:] - xys[:-1], dim=-1)

            # Minimize the total path length (for points not in an obstacle).
            dist_loss = norms.sum()

            # Spacing loss, to try to make the waypoints spaced evenly.
            # The idea here is that the norms on either side of any given
            # point should be roughly equal.
            spacing_loss = ((norms[1:] - norms[:-1]) ** 2).sum()

            losses: dict[str, Tensor] = {
                "dist": dist_loss * self.dist_loss_weight,
                "spacing": spacing_loss * self.spacing_loss_weight,
                "occ": occ_loss * self.occ_loss_weight,
            }

            # Embedding loss for the final waypoint; try to maximize the
            # similarity between it's clip embedding and the target embedding.
            if target_emb is not None:
                #sim_score_min = clip_preds_min @ target_emb.T
                #sim_score_max = clip_preds_max @ target_emb.T
                sim_score_min = torch.cosine_similarity(clip_preds_min, target_emb.T)
                sim_score_mid = torch.cosine_similarity(clip_preds_mid, target_emb.T)
                sim_score_max = torch.cosine_similarity(clip_preds_max, target_emb.T)
                sim_score = torch.cat([sim_score_min, sim_score_max]).max()
                #sim_score = torch.cat([sim_score_min, sim_score_mid, sim_score_max]).max()
                # sim_score = torch.where(is_inside_obs[-1], torch.zeros_like(sim_score), sim_score)
                sim_loss = -sim_score
                losses["sim"] = sim_loss * self.sim_loss_weight

            return losses

        # Optimization loop, just using gradient descent.
        prev_xys: Tensor | None = None

        #opt = torch.optim.Adam([xys], lr=self.lr)
        opt = torch.optim.SGD([xys], lr=self.lr, momentum=0.9)

        for _ in tqdm.trange(self.num_optimization_steps):
            opt.zero_grad()
            losses = get_losses(xys)
            loss = cast(Tensor, sum(losses.values()))
            loss.backward()

            # First point doesn't change, last point only changes if we have
            # a goal embedding rather than XY coordinate.
            #print(xys)
            xys_grad = cast(Tensor, xys.grad)
            xys_grad[:1].zero_()
            if target_emb is None:
                xys_grad[-1:].zero_()

            opt.step()

            # Check for NaN or Inf.
            if not torch.isfinite(xys).all():
                raise RuntimeError("Invalid waypoints.")

            # Termination conditions.
            is_finished = prev_xys is not None and torch.allclose(xys, prev_xys, atol=self.min_distance)
            if is_finished:
                break

            # if prev_xys is not None and i % 100 == 0:
            #     tqdm.tqdm.write(f"Losses: {losses}")
            #     tqdm.tqdm.write(f"diff: {(xys - prev_xys).norm().item()}")
            prev_xys = xys.detach().clone()

        points = [(x, y) for x, y in xys.detach().cpu().numpy().tolist()]  # pylint: disable=unnecessary-comprehension

        # Explicitly delete the optimizer and points.
        del opt, xys

        return points

    def is_valid_starting_point(self, xy: tuple[float, float]) -> bool:
        min_xyz, max_xyz = (xy[0], xy[1], self.min_z), (xy[0], xy[1], self.max_z)
        _, sdfs = self.get_clip_and_sdf([min_xyz, max_xyz])
        return all(sdf > self.occ_avoid_radius for sdf in sdfs)
