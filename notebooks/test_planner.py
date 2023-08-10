import os
from pathlib import Path

import ml.api as ml

from usa.models.point2emb import Point2EmbModel
from usa.planners.clip_sdf import AStarPlanner, GradientPlanner
from usa.tasks.clip_sdf import ClipSdfTask

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

cfg_path = Path(__file__).parent.resolve() / "runs" / "jupyter" / "run_0" / "config.yaml"

model, task = ml.load_model_and_task(cfg_path)
assert isinstance(model, Point2EmbModel)
assert isinstance(task, ClipSdfTask)


grid_planner = AStarPlanner(
    dataset=task._dataset(),
    model=model.double(),
    task=task.double(),
    device=task._device,
    # The heuristic to use for AStar
    heuristic="euclidean",
    # The grid resolution
    resolution=0.1,
    # Where to store cache artifacts
    cache_dir=None,
).double()

gradient_planner = GradientPlanner(
    dataset=task._dataset(),
    model=model.double(),
    task=task.double(),
    device=task._device,
    # The learning rate for the optimizer for the waypoints
    lr=1e-2,
    # The weight for the total path distance loss term
    dist_loss_weight=1.0,
    # The weight for the inter-point distance loss term
    spacing_loss_weight=1.0,
    # The weight for the "no-crashing-into-a-wall" loss term
    occ_loss_weight=25.0,
    # The weight for the loss term of the final semantic location
    sim_loss_weight=15.0,
    # Maximum number of optimization steps
    num_optimization_steps=1000,
    # If points move less than this distance, stop optimizing
    min_distance=1e-5,
    # Where to store cache artifacts
    # cache_dir=Path("cache"),
    cache_dir=None,
    # Height of the floor
    floor_height=0.1,
    # Height of the ceiling
    ceil_height=2.5,
).double()
