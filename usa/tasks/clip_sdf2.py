import functools
import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Sized, cast

import ml.api as ml
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as V
import tqdm
from omegaconf import MISSING
from torch import Tensor
from torch.utils.data.dataset import Dataset

from usa.models.clip import (
    CLIPTokenizer,
    cast_pretrained_model_key as cast_pretrained_clip_model_key,
    load_pretrained as load_pretrained_clip,
)
from usa.models.point2emb import Point2EmbModel
from usa.tasks.datasets.posed_rgbd import Bounds, get_posed_rgbd_dataset
from usa.tasks.datasets.types import PosedRGBDItem
from usa.tasks.datasets.utils import (
    aminmax,
    get_nearest_xyz,
    get_xy_pixel_from_xyz,
    get_xyz_coordinates,
)

import wandb

logger = logging.getLogger(__name__)

def clip_sim(a: Tensor, b: Tensor) -> Tensor:
    assert a.dim() == 2 and b.dim() == 2
    a, b = a / (a.norm(dim=1, keepdim=True) + 1e-3), b / (b.norm(dim=1, keepdim=True) + 1e-3)
    return a @ b.t()

@dataclass
class ClipSdfTaskConfig(ml.SupervisedLearningTaskConfig):
    dataset: str = ml.conf_field(MISSING, help="Dataset key to use")
    dataset_path: str | None = ml.conf_field(None, help="Path to the dataset")
    clip_model: str = ml.conf_field(MISSING, help="The CLIP model to load")
    queries: list[str] = ml.conf_field(MISSING, help="Queries to evaluate against")
    rotate_image: bool = ml.conf_field(False, help="If set, rotate image when getting CLIP scores")
    pts_per_frame: int = ml.conf_field(100, help="Number of points to sample per frame")
    min_depth_prct: float = ml.conf_field(0.1, help="Minimum depth percentage to sample")
    points_to_sample: int = ml.conf_field(50, help="XYZ points to sample (per frame) during model inference")
    image_shape: tuple[int, int] = ml.conf_field(lambda: (224, 224), help="Size of the SDF image to log")
    scales: list = ml.conf_field(lambda: [0.1, 0.2, 0.3, 0.4, 0.5], help="Scales of image crops")


class ClipModel:
    def __init__(self, key: str) -> None:
        """Wrapper for the CLIP model which isn't an nn.Module.

        This is useful because we don't want to save the CLIP model weights
        in every checkpoint, as they are not being trained, so it would be a
        waste of memory.

        Args:
            key: The key of the CLIP model to use.
        """

        with ml.Timer("loading pretrained CLIP model"):
            clip = load_pretrained_clip(cast_pretrained_clip_model_key(key), mode="all")
        self.visual = clip.visual
        self.linguistic = clip.linguistic

        # Disables gradients for CLIP model.
        self.visual.requires_grad_(False)
        self.linguistic.requires_grad_(False)

        self.tokenizer = CLIPTokenizer()


Model = Point2EmbModel
Batch = tuple
Output = tuple[Tensor, Tensor]
Loss = dict[str, Tensor]


@ml.register_task("ms_clip_sdf", ClipSdfTaskConfig)
class ClipSdfTask(ml.SupervisedLearningTask[ClipSdfTaskConfig, Model, Batch, Output, Loss]):
    bounds: Tensor
    pose_bounds: Tensor
    clip_text_embs: Tensor
    clip: ClipModel

    def __init__(self, config: ClipSdfTaskConfig) -> None:
        super().__init__(config)

        self.register_buffer("bounds", torch.zeros(3, 2))
        self.register_buffer("pose_bounds", torch.zeros(3, 2))

        self.clip = ClipModel(config.clip_model)
        self.dataset = self._get_posed_rgb_dataset()

    def _apply(self, fn: Any) -> Any:
        self.clip.visual._apply(fn)
        self.clip.linguistic._apply(fn)
        return super()._apply(fn)

    def apply(self, fn: Callable[["torch.nn.Module"], None]) -> Any:
        self.clip.visual.apply(fn)
        self.clip.linguistic.apply(fn)
        return super().apply(fn)

    def _get_posed_rgb_dataset(self) -> Dataset:
        return get_posed_rgbd_dataset(self.config.dataset, path=self.config.dataset_path)

    @functools.cached_property
    def _dataset(self) -> Dataset:
        return self.dataset
        #return get_posed_rgbd_dataset(self.config.dataset, path=self.config.dataset_path)

    @functools.lru_cache
    def text_clip_embs(self, device: torch.device) -> Tensor:
        with torch.no_grad():
            tokens = self.clip.tokenizer.tokenize(self.config.queries).to(device)
            embs = self.clip.linguistic(tokens)
            return embs

    @property
    def slice_height(self) -> float:
        return (self.pose_bounds[1, 0] + self.pose_bounds[1, 1]).item() / 2

    def run_model(self, model: Model, batch: Batch, state: ml.State) -> tuple[Tensor, Tensor, Tensor]:
        _, depth, mask, intrinsics, pose, _, _, scales = batch
        scales = scales[0]
        depth_frac = torch.rand_like(depth) * (1 - self.config.min_depth_prct) + self.config.min_depth_prct

        # Sample XYZ points to run through the model.
        sample_mask = torch.rand_like(mask, dtype=depth.dtype)
        q = torch.quantile(
            (sample_mask.cpu() if mask.device.type == "mps" else sample_mask).float(),
            (mask.shape[0] * self.config.points_to_sample) / sample_mask.numel(),
        ).to(sample_mask)
        mask = mask | (sample_mask > q)
        sampled_xyz = get_xyz_coordinates(depth_frac * depth, mask, pose, intrinsics).to(depth)

        num_scales = scales.shape[0]
        xyz_scales = (torch.ones((*sampled_xyz.shape[:-1], 1)) * scales.cpu()[torch.randperm(num_scales)[0]]).to(depth)

        sampled_xyz = sampled_xyz
        preds = model(sampled_xyz, xyz_scales)

        return preds, sampled_xyz, xyz_scales.flatten()[0]

    def compute_loss(self, model: Model, batch: Batch, state: ml.State, output: Output) -> Loss:
        rgb, depth, mask, intrinsics, pose, features, crop_centers, scales = batch
        scales = scales[0]
        preds, xyz, scale = output
        device = preds.device

        # Splits into CLIP and SDF predictions.
        clip_preds, sdf_preds = torch.split(preds, [self.clip.visual.output_dim, 1], dim=-1)

        with torch.no_grad():
            # Gets the nearest neighbor to the sampled points.
            batch_xyz = get_xyz_coordinates(depth.to(pose), mask, pose, intrinsics)
            nearest_xyz, nearest_inds = get_nearest_xyz(batch_xyz, xyz.to(pose))
            #if torch.any(torch.isnan(nearest_xyz)):
            #    print("\n\n\nhello nearest_xyz\n\n\n\n\n")
            #if torch.any(torch.isnan(nearest_inds)):
            #    print("\n\n\nhello nearest_inds\n\n\n\n\n")

            # Gets the batch ID of the nearest points.
            batch_inds = torch.arange(mask.shape[0], device=mask.device)
            batch_inds = batch_inds[:, None, None, None].expand_as(mask)[~mask][nearest_inds]

            # Gets the pixel closest to the nearest XYZ point.
            nearest_xy = get_xy_pixel_from_xyz(nearest_xyz, pose[batch_inds], intrinsics[batch_inds])
            #if torch.any(torch.isnan(nearest_xy)):
            #    print("\n\n\nhello nearest_xy1\n\n\n\n\n")
            nearest_xy = nearest_xy.round().int()
            #if torch.any(torch.isnan(nearest_xy)):
            #    print("\n\n\nhello nearest_xy2\n\n\n\n\n")
            scale_id = torch.where(scales == scale)[0].item()
            crop_center = crop_centers[scale_id].reshape(-1, 2)
            feature = features[scale_id]
            feature = feature.reshape(-1, feature.shape[-1])
            values, indices = torch.topk( \
                    torch.norm((nearest_xy.reshape(-1, 2).unsqueeze(1) - crop_center.unsqueeze(0)).float(), \
                    dim = 2), k = 4, largest = False)
            #if torch.any(torch.isnan(values)):
            #    print("\n\n\nhello values\n\n\n\n\n")
            #if torch.any(torch.isnan(indices)):
            #    print("\n\n\nhello indices\n\n\n\n\n")
            clip = torch.nn.functional.normalize(\
                    torch.sum(feature[indices] / (values.unsqueeze(-1) + 1e-3), dim = -2),\
                     p = 2, dim = -1)
            #if torch.any(torch.isnan(clip)):
            #    print("\n\n\nhello clip\n\n\n\n\n")
            #crop_result = get_image_crop_around(nearest_xy, rgb[batch_inds], (224, 224))

            # If there was a cropped image, get the CLIP embeddings for it.
            #crop_image: Tensor | None = None
            #clip: Tensor | None = None
            #if crop_result is not None:
            #    crop_image = crop_result[0]
            #    if self.config.rotate_image:
            #        crop_image = V.rotate(crop_image, -90)
            #    clip = self.clip.visual(crop_image)

            # Gets the CLIP embeddings for the cropped images.
            #clip = None if crop_result is None else self.clip.visual(crop_result[0])

            # Gets the SDF values.
            sdf = torch.norm(nearest_xyz - xyz, p=2, dim=1)

        # Updates the bounds and pose bounds.
        self.bounds.copy_(Bounds.merge_bounds(self.bounds, Bounds.from_xyz(batch_xyz)))
        self.pose_bounds.copy_(Bounds.merge_bounds(self.pose_bounds, Bounds.from_xyz(pose[..., :3, 3])))

        # Always compute the SDF loss since we have the nearest neighbor.
        losses = {"sdf": F.mse_loss(sdf_preds.squeeze(1), sdf.to(sdf_preds), reduction="none")}

        # Adds the CLIP loss if there were any cropped images.
        if clip is not None:
            sims = clip_sim(clip_preds, clip)
            clip_loss = F.cross_entropy(sims, torch.arange(sims.shape[0], device=device), reduction="none")

            # Weight the CLIP loss by the softmax of the SDF.
            clip_sdfs = sdf
            clip_loss = clip_loss * F.softmax(clip_sdfs, dim=0)

            losses["clip"] = clip_loss

        # Logs the prediction ranges.
        pmin, pmax = aminmax(preds)
        self.logger.log_scalar("pmin", pmin)
        self.logger.log_scalar("pmax", pmax)

        if state.phase == "valid":
            if wandb.run is not None:
                wandb.log({"sdf": torch.sum(losses["sdf"]).item()}, step = state.num_steps)
            if "clip" in losses and wandb.run is not None:
                wandb.log({"clip": torch.sum(losses["clip"]).item()}, step = state.num_steps)
            clip_images, sdf_image = self.get_clip_and_sdf_images(model, device, preds.dtype)
            #for i in range(len(clip_images)):
            #    if wandb.run is not None:
            #        wandb.log({self.config.queries[i] + " score map": wandb.Image(
            #                clip_images[i])
            #              }, step = state.num_steps)
            if wandb.run is not None:
                wandb.log({"sdf prediction": wandb.Image(
                            sdf_image)
                          }, step = state.num_steps)
            # clip_image, sdf_image = self.get_clip_and_sdf_images(model, device, preds.dtype)
            # self.logger.log_image("sdf", sdf_image)
            # self.logger.log_point_cloud("xyz", torch.stack([xyz, nearest_xyz.to(xyz)]))
            # self.logger.log_point_cloud("surface", batch_xyz.to(xyz))
            # self.logger.log_labeled_images("query_sims", (clip_image, self.config.queries))

            # if crop_image is not None:
            #     self.logger.log_images("cropped", crop_image)

        return losses

    def get_clip_and_sdf_images(
        self,
        model: ml.BaseModel,
        device: torch.device,
        dtype: torch.dtype,
        image_chunk_size: int = 128,
        scale: int = 120
    ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            bounds = Bounds.from_arr(self.bounds)
            pose_bounds = Bounds.from_arr(self.pose_bounds)
            slice_z = (pose_bounds.zmax + pose_bounds.zmin) / 2

            # Gets the XYZ points for the SDF image.
            width, height = self.config.image_shape
            xs, ys = torch.meshgrid(
                torch.linspace(bounds.xmin, bounds.xmax, width, device=device, dtype=dtype),
                torch.linspace(bounds.ymin, bounds.ymax, height, device=device, dtype=dtype),
            )
            xyz = torch.stack((xs, ys, torch.full_like(xs, slice_z)), dim=-1).flatten(0, -2)
            num_chunks = xyz.shape[0] // image_chunk_size

            # Compute CLIP similarity with the text embeddings.
            text_embs = self.text_clip_embs(device)

            # Runs all XYZ points through the model to get predictions.
            all_clip_preds: list[Tensor] = []
            all_sdf_preds: list[Tensor] = []
            for xyz_chunk in tqdm.tqdm(torch.chunk(xyz, num_chunks), disable=ml.is_distributed()):
                preds = model(xyz_chunk, torch.ones(*xyz_chunk.shape[:-1], 1).to(device) * scale)
                clip_preds, sdf_preds = torch.split(preds, [self.clip.visual.output_dim, 1], dim=-1)
                all_clip_preds.append(clip_sim(clip_preds, text_embs))
                all_sdf_preds.append(sdf_preds)

            # Concatenates the predictions and converts them to image shapes.
            sdf_preds = torch.cat(all_sdf_preds, dim=0).view(width, height)
            clip_preds = torch.cat(all_clip_preds, dim=0).view(width, height, text_embs.shape[0])

            # Normalizes CLIP similarity image to range (0, 1).
            clip_preds = clip_preds.softmax(dim=-1)
            clip_min, clip_max = aminmax(clip_preds)
            clip_preds = (clip_preds - clip_min) / (clip_max - clip_min + 1e-3)

            # CLIP image has shape (num_embs, width, height)
            # SDF image has shape (width, height)
            return clip_preds.permute(2, 0, 1), sdf_preds

    def get_dataset(self, phase: ml.Phase) -> Dataset[PosedRGBDItem]:
        return self._dataset


def test_sdf_dataset(max_samples: int = 3) -> None:
    """Provides a helper script for testing the SDF dataset.

    Usage:
        python -m ml.tasks.sdf

    Args:
        max_samples: Maximum number of samples to test
    """

    ml.configure_logging(use_tqdm=True)

    config = ClipSdfTaskConfig(dataset="replica_apt_2_mnp")
    task = ClipSdfTask(config)
    ds = task.get_dataset("train")

    for i in itertools.islice(tqdm.trange(len(cast(Sized, ds)), total=max_samples, desc="Samples"), max_samples):
        logger.info("Checking sample %d", i)
        ds[i].check()


if __name__ == "__main__":
    test_sdf_dataset()
