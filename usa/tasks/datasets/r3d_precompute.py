import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import cv2
import liblzfse
import ml.api as ml
import numpy as np
import torch
import torchvision.transforms.functional as V
from PIL import Image
from quaternion import as_rotation_matrix, quaternion
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from usa.tasks.datasets.types import PosedRGBDItem
from usa.tasks.datasets.utils import visualize_posed_rgbd_dataset
import clip
from torchvision import transforms

logger = logging.getLogger(__name__)

LAB_DATASET_URL = "https://github.com/codekansas/usa/releases/download/v0.0.2/lab.r3d"
STUDIO_DATASET_URL = "https://github.com/codekansas/usa/releases/download/v0.0.2/studio.r3d"


@dataclass(frozen=True)
class Metadata:
    rgb_shape: tuple[int, int]
    depth_shape: tuple[int, int]
    fps: int
    timestamps: np.ndarray  # (T) the camera frame timestamps
    intrinsics: np.ndarray  # (3, 3) intrinsics matrix
    poses: np.ndarray  # (T, 4, 4) camera pose matrices
    start_pose: np.ndarray  # (4, 4) initial camera pose matrix


def as_pose_matrix(pose: list[float]) -> np.ndarray:
    """Converts a list of pose parameters to a pose matrix.

    Args:
        pose: The list of pose parameters, (qx, qy, qz, qw, px, py, pz)

    Returns:
        A (4, 4) pose matrix
    """

    mat = np.eye(4, dtype=np.float64)
    qx, qy, qz, qw, px, py, pz = pose
    mat[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
    mat[:3, 3] = [px, py, pz]
    return mat


def get_arrs(
    r3d_file: ZipFile,
    meta: Metadata,
    use_depth_shape: bool = True,
    subsample_rate: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads the arrays from the .r3d file.

    Args:
        r3d_file: The open .r3d file
        meta: The metadata loaded from the file
        use_depth_shape: If True, the image array will be resized to the depth
            shape; otherwise, the depth array will be resized to the image
            shape.

    Returns:
        The images array, with shape (T, H, W, 3); the depth array, with
        shape (T, H, W), and the mask array, with shape (T, H, W) and
        boolean values where True means that the point is masked. Note that
        the depth and confidence arrays are resized to match the image array
        shape.
    """

    logger.info("Preprocessing R3D arrays")
    img_re_expr = re.compile(r"^rgbd/(\d+).jpg$")
    depth_re_expr = re.compile(r"^rgbd/(\d+).depth$")
    conf_re_expr = re.compile(r"^rgbd/(\d+).conf$")

    tsz = meta.timestamps.shape[0]
    rgb_h, rgb_w = meta.rgb_shape
    depth_h, depth_w = meta.depth_shape

    def get_filenames(expr: re.Pattern) -> list[str]:
        re_matches = sorted(
            [re_match for re_match in (expr.match(f.filename) for f in r3d_file.filelist) if re_match is not None],
            key=lambda m: int(m.group(1)),
        )
        return [m.group() for m in re_matches]

    def to_img_shape(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(arr, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

    def to_depth_shape(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(arr, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)

    img_fnames = get_filenames(img_re_expr)
    depth_fnames = get_filenames(depth_re_expr)
    conf_fnames = get_filenames(conf_re_expr)

    assert img_fnames[0] == "rgbd/0.jpg" and img_fnames[-1] == f"rgbd/{len(img_fnames) - 1}.jpg"
    assert len(img_fnames) == tsz
    assert len(depth_fnames) == tsz
    assert len(conf_fnames) == tsz

    arr_h, arr_w = (depth_h, depth_w) if use_depth_shape else (rgb_h, rgb_w)
    img_arrs = np.zeros((tsz, arr_h, arr_w, 3), dtype=np.uint8)
    depth_arrs = np.zeros((tsz, arr_h, arr_w), dtype=np.float32)
    conf_arrs = np.zeros((tsz, arr_h, arr_w), dtype=np.uint8)

    zipped = zip(img_fnames, depth_fnames, conf_fnames)
    iterable = enumerate(tqdm(zipped, total=tsz, desc="Loading R3D file", disable=ml.is_distributed()))
    for i, (img_fname, depth_fname, conf_fname) in iterable:
        with r3d_file.open(img_fname, "r") as img_f:
            img_arr = np.asarray(Image.open(img_f))
        assert img_arr.shape == (rgb_h, rgb_w, 3)
        img_arrs[i] = to_depth_shape(img_arr) if use_depth_shape else img_arr

        with r3d_file.open(depth_fname, "r") as depth_f:
            raw_bytes = depth_f.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_arr = np.frombuffer(decompressed_bytes, dtype=np.float32).reshape(depth_h, depth_w).copy()
        depth_is_nan_arr = np.isnan(depth_arr)
        depth_arr[depth_is_nan_arr] = -1.0
        depth_arrs[i] = depth_arr if use_depth_shape else to_img_shape(depth_arr)

        with r3d_file.open(conf_fname, "r") as conf_f:
            raw_bytes = conf_f.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            conf_arr = np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape(depth_h, depth_w).copy()
        conf_arr[depth_is_nan_arr] = 0
        conf_arrs[i] = conf_arr if use_depth_shape else to_img_shape(conf_arr)

    masks_arrs = conf_arrs != 2

    img_arrs = img_arrs[::subsample_rate]
    depth_arrs = depth_arrs[::subsample_rate]
    masks_arrs = masks_arrs[::subsample_rate]
    return img_arrs, depth_arrs, masks_arrs


def read_metadata(r3d_file: ZipFile, use_depth_shape: bool) -> Metadata:
    with r3d_file.open("metadata", "r") as f:
        metadata_dict = json.load(f)

    metadata = Metadata(
        rgb_shape=(metadata_dict["h"], metadata_dict["w"]),
        depth_shape=(metadata_dict["dh"], metadata_dict["dw"]),
        fps=metadata_dict["fps"],
        timestamps=np.array(metadata_dict["frameTimestamps"], dtype=np.float64),
        intrinsics=np.array(metadata_dict["K"], dtype=np.float64).reshape(3, 3).T,
        poses=np.stack([as_pose_matrix(pose) for pose in metadata_dict["poses"]], axis=0),
        start_pose=as_pose_matrix(metadata_dict["initPose"]),
    )

    # Converts the intrinsics from the image shape to the depth shape.
    if use_depth_shape:
        metadata.intrinsics[0, :] *= metadata.depth_shape[1] / metadata.rgb_shape[1]
        metadata.intrinsics[1, :] *= metadata.depth_shape[0] / metadata.rgb_shape[0]

    # Swaps fx and cx, fy and cy in the intrinsics.
    # metadata.intrinsics[0, 0], metadata.intrinsics[1, 1] = metadata.intrinsics[1, 1], metadata.intrinsics[0, 0]
    # metadata.intrinsics[0, 2], metadata.intrinsics[1, 2] = metadata.intrinsics[1, 2], metadata.intrinsics[0, 2]

    # Checks metadata is well-formed.
    assert metadata.timestamps.shape[0] == metadata.poses.shape[0]
    assert metadata.poses.shape[1:] == (4, 4)
    assert metadata.start_pose.shape == (4, 4)

    return metadata


class MS_R3DDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        img_dim: int | None = None,
        random_crop: bool = True,
        use_depth_shape: bool = True,
        scales = torch.arange(30, 191, 40)
    ) -> None:
        """Defines a dataset for iterating samples from an R3D file.

        The .r3d file format is the special format used by the Record3D app.
        It is basically just a zipped file with some images, depths and
        metadata.

        Args:
            path: The path to the .r3d file
            img_dim: If set, crop image to a square with this image shape
            random_crop: If `img_dim` is set and the image has to be cropped,
                use this option to toggle random verses center cropping
            use_depth_shape: If True, the image array will be resized to the depth
                shape; otherwise, the depth array will be resized to the image
                shape.
        """

        path = Path(path)

        assert path.suffix == ".r3d", f"Invalid file suffix: {path.suffix} Expected `.r3d`"

        self.use_depth_shape = use_depth_shape

        with ZipFile(path) as r3d_file:
            self.metadata = read_metadata(r3d_file, self.use_depth_shape)
            self.imgs_arr, self.depths_arr, self.masks_arr = get_arrs(r3d_file, self.metadata, self.use_depth_shape)

        self.img_dim = img_dim
        self.random_crop = random_crop
        self.intrinsics = self.metadata.intrinsics
        self.scales = scales
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.clip_model, self.preprocessor = clip.load("ViT-B/32", device = self.device)
        self.clip_model.eval()
        self.features, self.crop_centers = self._get_features()
        del self.clip_model
        affine_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
        self.poses = self.metadata.poses @ affine_matrix

        # Converts poses from (X, Z, Y) to (X, -Y, Z).
        affine_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        self.poses = affine_matrix @ self.poses

    def _get_features(self):
        all_features = []
        all_crop_centers = []

        for index in tqdm(range(len(self.imgs_arr)), total=len(self.imgs_arr), desc="Computing features"):
            features = []
            img = torch.from_numpy(self.imgs_arr[index]).permute(2, 0, 1)
            img = V.convert_image_dtype(img, torch.float32)
            depth = torch.from_numpy(self.depths_arr[index]).unsqueeze(0)
            mask = torch.from_numpy(self.masks_arr[index]).unsqueeze(0)

            #if self.img_dim is not None:
            #    if self.random_crop:
            #        crop_shape = [self.img_dim, self.img_dim]
            #        img, depth, mask = ml.transforms.random_square_crop_multi([img, depth, mask])
            #        img, depth, mask = V.resize(img, crop_shape), V.resize(depth, crop_shape), V.resize(mask, crop_shape)
            #    else:
            #        img = ml.transforms.square_resize_crop(img, self.img_dim)
            #        depth = ml.transforms.square_resize_crop(depth, self.img_dim)
            #        mask = ml.transforms.square_resize_crop(mask, self.img_dim)
            
            C, W, H = img.shape
            for ind, scale in enumerate(self.scales):
                crop_size = scale.item()
                stride = int(crop_size * (1 - 0.5))
                unfold_layer = torch.nn.Unfold(kernel_size=crop_size, stride=stride)
                #print(crop_size, stride, img.shape, img)
                #print(unfold_layer(torch.randn(3, 256, 192)).unsqueeze(0))
                unfolded = unfold_layer(img.unsqueeze(0))
                resize = transforms.Resize((224, 224))
                crops = resize(unfolded.view(3, crop_size, crop_size, -1).permute(3, 0, 1, 2)).to(self.device)
                with torch.no_grad():
                    embeddings = self.clip_model.encode_image(crops)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1).detach().cpu()
                features.append(embeddings)
            all_features.append(features)
        
        for ind, scale in enumerate(self.scales):
            crop_size = scale.item()
            stride = int(crop_size * (1 - 0.5))
            num_patches_height = (H - crop_size) // stride + 1
            num_patches_width = (W - crop_size) // stride + 1
            x_coords = torch.arange(0, num_patches_width) * stride + crop_size // 2
            y_coords = torch.arange(0, num_patches_height) * stride + crop_size // 2
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
            crop_center = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
            all_crop_centers.append(crop_center)

        return all_features, all_crop_centers

    def __len__(self) -> int:
        #return self.metadata.timestamps.shape[0]
        return len(self.imgs_arr)

    def __getitem__(self, index: int) -> PosedRGBDItem:
        img = torch.from_numpy(self.imgs_arr[index]).permute(2, 0, 1)
        img = V.convert_image_dtype(img, torch.float32)
        depth = torch.from_numpy(self.depths_arr[index]).unsqueeze(0)
        mask = torch.from_numpy(self.masks_arr[index]).unsqueeze(0)
        intr = torch.from_numpy(self.intrinsics)
        pose = torch.from_numpy(self.poses[index])

        #if self.img_dim is not None:
        #    if self.random_crop:
        #        crop_shape = [self.img_dim, self.img_dim]
        #        img, depth, mask = ml.transforms.random_square_crop_multi([img, depth, mask])
        #        img, depth, mask = V.resize(img, crop_shape), V.resize(depth, crop_shape), V.resize(mask, crop_shape)
        #    else:
        #        img = ml.transforms.square_resize_crop(img, self.img_dim)
        #        depth = ml.transforms.square_resize_crop(depth, self.img_dim)
        #        mask = ml.transforms.square_resize_crop(mask, self.img_dim)

            # Intrinsics correction not implemented yet.
            #intr.fill_(-1.0)

        #item = PosedRGBDItem(image=img, depth=depth, mask=mask, intrinsics=intr, pose=pose)
        if ml.is_debugging():
            item.check()
        return img, depth, mask, intr, pose, self.features[index], self.crop_centers, self.scales
