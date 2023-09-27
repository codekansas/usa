import ml as ml_root
print(ml_root.__version__)

import os

# These environment variables control where training and eval logs are written.
# You can set these in your shell profile as well.
os.environ["RUN_DIR"] = "runs"
os.environ["EVAL_RUN_DIR"] = "eval_runs"
os.environ["MODEL_DIR"] = "models"
os.environ["DATA_DIR"] = "data"

# This is used to set a constant Tensorboard port.
os.environ["TENSORBOARD_PORT"] = str(8989)

# Useful for debugging.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import ml.api as ml  # Source: https://github.com/codekansas/ml-starter

ml.configure_logging()

# Imports these files to add them to the model and task registry.
from usa.models.point2emb import Point2EmbModel, Point2EmbModelConfig
from usa.tasks.clip_sdf import ClipSdfTask

import math
import pickle as pkl
import zipfile
from pathlib import Path
from typing import Iterator

import cv2
import imageio
import matplotlib.pyplot as plt
import ml.api as ml
import numpy as np
import requests
import torch
from IPython.display import Image
from omegaconf import OmegaConf
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from torch import Tensor

import hydra
import wandb
wandb.init(project="usa-net")

@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(config):
    # Using the default config, but overriding the dataset.
    #config = OmegaConf.load("usa/notebooks/config.yaml")
    #config.task.dataset = "r3d"
    #config.task.dataset_path = "clip-fields/Kitchen.r3d"

    # We still need to explicitly set these variables.
    #config.trainer.exp_name = "4_256_no"
    #config.trainer.base_run_dir = "nyu_kitchen"
    #config.trainer.run_id = 0

    # Only use stdout logger.
    #config.logger = [{"name": "stdout"}]
    #config.task.dataloader.train.batch_size = 16

    # You can change this number to change the number of training steps.
    #config.task.finished.max_steps = 8000

    # Loads the config objects.
    objs = ml.instantiate_config(config)

    # Unpacking the different components.
    model = objs.model
    task = objs.task
    optimizer = objs.optimizer
    lr_scheduler = objs.lr_scheduler
    trainer = objs.trainer

    # Runs the training loop.
    trainer.train(model, task, optimizer, lr_scheduler)

if __name__ == "__main__":
    main()
