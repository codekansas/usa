import os

# These environment variables control where training and eval logs are written.
# You can set these in your shell profile as well.
os.environ["RUN_DIR"] = "runs"
os.environ["EVAL_RUN_DIR"] = "eval_runs"
os.environ["MODEL_DIR"] = "models"
os.environ["DATA_DIR"] = "data"

# This is used to set a constant Tensorboard port.
os.environ["TENSORBOARD_PORT"] = str(8989)

import ml.api as ml  # Source: https://github.com/codekansas/ml-starter

# Imports these files to add them to the model and task registry.
from usa.models.point2emb import Point2EmbModel
from usa.tasks.clip_sdf import ClipSdfTask

ml.configure_logging(use_tqdm=True)

config = {
    "model": {
        "name": "point2emb",          # `register_model` name in `usa.models.point2emb`
        "num_layers": 4,
        "hidden_dims": 256,
        "output_dims": 513,           # CLIP = 512, SDF = 1
        "num_pos_embs": 6,
        "norm": "no_norm",
        "act": "relu"
    },
    "task": {
        "name": "clip_sdf",           # `register_task` name in `usa.tasks.clip_sdf`
        "dataset": "lab_r3d",         # Pre-collected dataset
        "clip_model": "ViT_B_16",
        "queries": [
            "Chair",
            "Shelves",
            "Man sitting at a computer",
            "Desktop computers",
            "Wooden box",
            "Doorway",
        ],
        "rotate_image": True,         # Dataset-specific, for visualization purposes
        "finished": {
            "max_steps": 10_000,      # Number of training steps
        },
        "dataloader": {
            "train": {
                "batch_size": 16,
                "num_workers": 0,
                "persistent_workers": False,
            },
        },
        "loss":{
            "reduce_type":"mean"
        }
    },
    "optimizer": {
        "name": "adam",
        "lr": 3e-4,
    },
    "lr_scheduler": {
        "name": "linear",
    },
    "trainer": {
        "detect_anomaly": False,
        "use_tf32": True,
        "deterministic": False,
        "name": "sl",
        "exp_name": "jupyter",
        "log_dir_name": "test",
        "base_run_dir": "runs",
        "run_id": 0,
        "checkpoint": {
            "save_every_n_steps": 2500,
            "only_save_most_recent": True,
        },
        "validation": {
            "valid_every_n_steps": 250,
            "num_init_valid_steps": 1,
        },
        "cpu_stats_ping_interval":{
            "cpu_stats_ping_interval": 1,
            "cpu_stats_only_log_once": False
        },
        "torch_compile":{
            "enabled": True,
            "init_scale": 2.0**16,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "backend": "auto",
            "fullgraph": True
        }
    },
    "logger": [{"name": "tensorboard",
               "start_in_subprocess": True,
               "log_id": "test1"}],
}

objs = ml.instantiate_config(config)

# Unpacking the different components.
model = objs.model
task = objs.task
optimizer = objs.optimizer
lr_scheduler = objs.lr_scheduler
trainer = objs.trainer

from tensorboard import notebook

# Show Tensorboard inside the notebook.
notebook.display(port=int(os.environ['TENSORBOARD_PORT']))

# Runs the training loop.
trainer.train(model, task, optimizer, lr_scheduler)