import typing as t
import warnings

import rich
from torch import Tensor

warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from jaxtyping import Float, Bool

from quick_start_helper import create_colored_pcd, _filter_by_conf
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

from dataclasses import dataclass


@dataclass
class VGGTReturnType:
    image_paths: t.List[Path]
    intrinsic: Float[np.ndarray, "batch 3 3"]
    c2w: Float[np.ndarray, "batch 4 4"]
    depths: Float[np.ndarray, "batch h w"]
    depth_conf: Float[np.ndarray, "batch h w"]
    points: Float[np.ndarray, "batch h w 3"]
    point_conf: Float[np.ndarray, "batch h w 3"]
    colored_pointcloud_from_pointmap: o3d.geometry.PointCloud
    colored_pointcloud_from_depthmap: o3d.geometry.PointCloud


@torch.no_grad()
@torch.cuda.amp.autocast(dtype=dtype)
def run_inference(image_paths: t.List[Path], conf_threshold: float = 0.001) -> VGGTReturnType:
    rich.print(image_paths)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    _URL = "/home/jizong/.cache/huggingface/hub/model.pt"
    model.load_state_dict(torch.load(_URL, map_location=device, weights_only=True))
    model.to(device)

    images = load_and_preprocess_images(image_paths).to(device)

    images = images[None]  # add batch dimension
    aggregated_tokens_list, ps_idx = model.aggregator(images)

    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # maybe extract the camera pose
    c2w = torch.zeros(extrinsic.shape[1], 4, 4)
    c2w[:, 3, 3] = 1.0
    c2w[:, :3, :] = extrinsic[0]
    c2w = c2w.inverse()
    c2w: Float[Tensor, "batch 4 4"]

    intrinsic_np:Float[np.ndarray, "batch 3 3"] = intrinsic[0].cpu().numpy()
    c2w_np: Float[np.ndarray, "batch 4 4"] = c2w.cpu().numpy()

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    depth_map: Float[Tensor, "1 batch h w 1"]
    depth_conf: Float[Tensor, "1 batch h w"]
    filtered_mask = _filter_by_conf(depth_conf, threshold=conf_threshold)
    filtered_mask: Bool[Tensor, "1 batch h w"]

    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0).cpu().numpy(),
                                                                 extrinsic.squeeze(0).cpu().numpy(),
                                                                 intrinsic.squeeze(0).cpu().numpy())
    point_map_by_unprojection: Float[np.ndarray, "batch h w 3"]

    pcd_from_depth = create_colored_pcd(
        xyz=torch.from_numpy(point_map_by_unprojection[None, ...]).cuda()[filtered_mask],
        pcd_xyz_dim=-1, color=images.moveaxis(2, -1)[filtered_mask],
        color_dim=-1)

    depth_map_np:Float[np.ndarray, "batch h w"] = depth_map[0].cpu().numpy()
    depth_conf_np:Float[np.ndarray, "batch h w"] = depth_conf[0].cpu().numpy()

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
    # visualize filtered
    filtered_mask = _filter_by_conf(point_conf, threshold=conf_threshold)

    pcd_from_pointmap = create_colored_pcd(xyz=point_map[filtered_mask], pcd_xyz_dim=-1,
                                           color=images.moveaxis(2, -1)[filtered_mask],
                                           color_dim=-1)

    point_map_np = point_map[0].cpu().numpy()
    point_conf_np = point_conf[0].cpu().numpy()


    result = VGGTReturnType(
        image_paths=image_paths,
        intrinsic=intrinsic_np,
        c2w=c2w_np,
        depths=depth_map_np,
        depth_conf=depth_conf_np,
        points=point_map_np,
        point_conf=point_conf_np,
        colored_pointcloud_from_pointmap=pcd_from_pointmap,
        colored_pointcloud_from_depthmap=pcd_from_depth,
    )
    return result


if __name__ == "__main__":
    res= run_inference(
        [
            Path("/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012346/0000000107.jpeg"),
            Path("/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012347/0000000107.jpeg"),
            Path("/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012348/0000000107.jpeg"),
            Path("/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN20230102350/0000000107.jpeg"),

            # Path(
            #     "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012346/0000000109.jpeg"),
            # Path(
            #     "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012347/0000000109.jpeg"),
            # Path(
            #     "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012348/0000000109.jpeg"),
            # Path(
            #     "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN20230102350/0000000109.jpeg"),


         ],
        conf_threshold=0.00
    )
    o3d.visualization.draw_geometries([res.colored_pointcloud_from_depthmap])