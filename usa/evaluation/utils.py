import os
from dataclasses import dataclass
from pathlib import Path
from typing import Type

import matplotlib.pyplot as plt
import ml.api as ml
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data.dataset import Dataset

from usa.models.point2emb import Point2EmbModel
from usa.planners.base import Map, Planner
from usa.planners.clip_sdf import (
    AStarPlanner as AStarClipSDFPlanner,
    GradientPlanner as GradientClipSDFPlanner,
)
from usa.planners.occupancy_map import (
    AStarPlanner as AStarOccupancyMapPlanner,
)
from usa.tasks.clip_sdf import ClipSdfTask
from usa.tasks.datasets.posed_rgbd import get_poses
from usa.tasks.datasets.types import PosedRGBDItem


@dataclass
class EvalSet:
    name: str
    dataset: Dataset[PosedRGBDItem]
    planners: dict[str, Planner]
    rotate: bool = False
    concat_horizontal: bool = False


def get_eval_root_dir() -> Path:
    root_dir = ml.get_eval_run_dir() / "usa"
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


def load_clip_sdf_model_from_ckpt_and_config(
    ckpt_path: Path,
    config_path: Path,
    device: Type[ml.BaseDevice],
) -> tuple[Point2EmbModel, ClipSdfTask]:
    """Loads the CLIP SDF model from the experiment path.

    Args:
        ckpt_path: The path to the checkpoint file.
        config_path: The path to the config file.
        device: The device to load the model on.

    Returns:
        The loaded model, task.
    """

    # Gets the model config.
    config = OmegaConf.load(config_path)
    model = Point2EmbModel(config.model)
    task = ClipSdfTask(config.task)

    # Loads the model and task checkpoints.
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    task.load_state_dict(ckpt["task"])

    device.module_to(model)
    device.module_to(task)

    return model, task


def load_clip_sdf_model(exp_path: Path, device: Type[ml.BaseDevice]) -> tuple[Point2EmbModel, ClipSdfTask]:
    """Loads the CLIP SDF model from the experiment path.

    Args:
        exp_path: The path to the experiment path (probably ending in something
            like "run_0/")
        device: The device to load the model on.

    Returns:
        The loaded model and task.

    Raises:
        ValueError: If the model could not be loaded.
    """

    if not exp_path.is_dir():
        raise ValueError(f"Expected {exp_path} to be a directory")

    # Gets the config path.
    config_path = exp_path / "config.yaml"

    # Gets the checkpoint path.
    if not (ckpt_path := exp_path / "checkpoints" / "ckpt.ckpt").is_file():
        ckpt_path = exp_path / "ckpt.pt"
    if not ckpt_path.is_file():
        raise ValueError(f"Could not find checkpoint file at {ckpt_path}")

    return load_clip_sdf_model_from_ckpt_and_config(ckpt_path, config_path, device)


def get_planners(
    name: str,
    use_occ_grid_planners: bool,
    floor_height: float = 0.1,
    ceil_height: float = 1.3,
) -> tuple[dict[str, Planner], Dataset[PosedRGBDItem], np.ndarray]:
    """Returns the planners for a given evaluation set.

    Args:
        name: The name of the evaluation set.
        use_occ_grid_planners: If the occupancy grid planners should be used.
        floor_height: The height of the floor.
        ceil_height: The height of the ceiling.

    Returns:
        A dictionary of planners to evaluate.

    Raises:
        EnvironmentError: If the environment is not configured correctly.
    """

    # Floor height corrections.
    if name in ("lab_r3d", "studio_r3d"):
        cam_height = 1.2
        floor_height, ceil_height = floor_height - cam_height, ceil_height - cam_height

    # Seed everything for reproducibility.
    ml.set_random_seed(1337)

    # Gets the directory where the checkpoint is saved.
    if "CKPTS_PATH_ROOT" in os.environ:
        clip_sdf_exp_path = Path(os.environ["CKPTS_PATH_ROOT"]) / name
    else:
        clip_sdf_exp_path = get_eval_root_dir() / "ckpts" / name
    if not clip_sdf_exp_path.is_dir():
        raise EnvironmentError(f"Checkpoint directory not found: {clip_sdf_exp_path}")

    # Gets the device (GPU, CPU, Metal...)
    os.environ["USE_FP64"] = "1"
    device = ml.AutoDevice.detect_device()

    model, task = load_clip_sdf_model(clip_sdf_exp_path, device)
    dataset = task.get_dataset("train")

    # Cache to the experiment directory.
    base_cache_dir = clip_sdf_exp_path / "eval_cache" / "semantics" / name

    planners: dict[str, Planner] = {
        "a_star_10_cm_clip_sdf": AStarClipSDFPlanner(
            dataset,
            model,
            task,
            device,
            "euclidean",
            resolution=0.1,
            cache_dir=base_cache_dir / "a_star_10_cm_clip_sdf",
            floor_height=floor_height,
            ceil_height=ceil_height,
        ),
        "a_star_20_cm_clip_sdf": AStarClipSDFPlanner(
            dataset,
            model,
            task,
            device,
            "euclidean",
            resolution=0.2,
            cache_dir=base_cache_dir / "a_star_20_cm_clip_sdf",
            floor_height=floor_height,
            ceil_height=ceil_height,
        ),
        "a_star_30_cm_clip_sdf": AStarClipSDFPlanner(
            dataset,
            model,
            task,
            device,
            "euclidean",
            resolution=0.3,
            cache_dir=base_cache_dir / "a_star_30_cm_clip_sdf",
            floor_height=floor_height,
            ceil_height=ceil_height,
        ),
        "a_star_40_cm_clip_sdf": AStarClipSDFPlanner(
            dataset,
            model,
            task,
            device,
            "euclidean",
            resolution=0.4,
            cache_dir=base_cache_dir / "a_star_40_cm_clip_sdf",
            floor_height=floor_height,
            ceil_height=ceil_height,
        ),
        "gradient_clip_sdf": GradientClipSDFPlanner(
            dataset,
            model,
            task,
            device,
            cache_dir=base_cache_dir / "gradient_clip_sdf",
            floor_height=floor_height,
            ceil_height=ceil_height,
        ),
    }

    if use_occ_grid_planners:
        planners.update(
            {
                "a_star_10_cm_occ_map": AStarOccupancyMapPlanner(
                    dataset,
                    "euclidean",
                    resolution=0.1,
                    cache_dir=base_cache_dir / "a_star_10_cm_occ_map",
                    floor_height=floor_height,
                    ceil_height=ceil_height,
                ),
                # "a_star_20_cm_occ_map": AStarOccupancyMapPlanner(
                #     dataset,
                #     "euclidean",
                #     resolution=0.2,
                #     cache_dir=base_cache_dir / "a_star_20_cm_occ_map",
                #     floor_height=floor_height,
                #     ceil_height=ceil_height,
                # ),
                # "a_star_30_cm_occ_map": AStarOccupancyMapPlanner(
                #     dataset,
                #     "euclidean",
                #     resolution=0.3,
                #     cache_dir=base_cache_dir / "a_star_30_cm_occ_map",
                #     floor_height=floor_height,
                #     ceil_height=ceil_height,
                # ),
                # "a_star_40_cm_occ_map": AStarOccupancyMapPlanner(
                #     dataset,
                #     "euclidean",
                #     resolution=0.4,
                #     cache_dir=base_cache_dir / "a_star_40_cm_occ_map",
                #     floor_height=floor_height,
                #     ceil_height=ceil_height,
                # ),
            }
        )

    poses = get_poses(dataset, base_cache_dir)

    return planners, dataset, poses


def plot_paths(
    planner_map: Map,
    paths: list[list[tuple[float, float]]],
    output_path: Path,
) -> None:
    """Plots a planner path and saves it to the output directory.

    Args:
        planner_map: The planner map.
        paths: The paths.
        output_path: The output directory.
    """

    # Converts the path to a numpy array.
    path_arrs = [np.array([planner_map.to_pt(p) for p in path]) for path in paths]

    # Gets the start and end points.
    start_pts = [p[0] for p in path_arrs]
    end_pts = [p[-1] for p in path_arrs]

    # Plots the path.
    plt.figure(figsize=(10, 10))
    plt.imshow(planner_map.grid, cmap="gray")
    for path_arr, start_pt, end_pt in zip(path_arrs, start_pts, end_pts):
        color = np.random.rand(3) if len(paths) > 1 else "red"

        # Plots line and waypoints.
        plt.plot(path_arr[:, 0], path_arr[:, 1], color=color)
        plt.scatter(path_arr[:, 0], path_arr[:, 1], color=color, s=5)

        plt.plot(start_pt[0], start_pt[1], "go")
        plt.plot(end_pt[0], end_pt[1], "ro")

    # Removes the axes and whitespace.
    plt.axis("off")
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_map(planner_map: Map, output_path: Path) -> None:
    """Plots a planner map and saves it to the output directory.

    Args:
        planner_map: The planner map.
        output_path: The output directory.
    """

    # Plots the path.
    plt.figure(figsize=(10, 10))
    plt.imshow(planner_map.grid, cmap="gray")

    # Removes the axes and whitespace.
    plt.axis("off")
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
