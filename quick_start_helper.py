import numpy as np
from jaxtyping import Float, Bool
from torch import Tensor
import open3d as o3d

def create_colored_pcd(*, xyz: Tensor, pcd_xyz_dim: int = -1, color: Tensor | None,
                       color_dim: int = -1) -> o3d.geometry.PointCloud:
    xyz = xyz.moveaxis(pcd_xyz_dim, -1)
    assert xyz.shape[-1] == 3, xyz.shape
    if color is not None:
        color=color.moveaxis(color_dim, -1)
        assert color.shape[-1] in (3, 4), color.shape
        if color.shape[-1] == 4:
            color = color[..., :3]

    flatten_xyz = xyz.reshape(-1, 3)
    if color is not None:
        flatten_color = color.reshape(-1, 3)
        assert flatten_xyz.shape[0] == flatten_color.shape[0], (flatten_xyz.shape, flatten_color.shape)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(flatten_xyz.detach().cpu().numpy())
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(flatten_color.detach().cpu().numpy())

    return pcd


def added_camera_center(point_cloud:o3d.geometry.PointCloud, camera_centers: Float[Tensor, "*b 3"]):
    assert camera_centers.shape[-1]==3, camera_centers.shape
    camera_centers_np = camera_centers.detach().cpu().numpy().reshape(-1, 3)

    xyz = np.array(point_cloud.points)
    color = np.array(point_cloud.colors)

    new_xyz = np.concatenate([xyz, camera_centers_np], axis=0)
    new_color = np.concatenate([color, np.zeros_like(camera_centers_np)], axis=0)
    new_color[-camera_centers_np.shape[0]:, 0] = 1.0
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_xyz)
    new_pcd.colors = o3d.utility.Vector3dVector(new_color)

    return new_pcd


def _filter_by_conf(point_conf: Float[Tensor, "*d 3"],
                    threshold: float = 0.5) -> Bool[Tensor, "*d 3"]:
    point_conf = (point_conf[...] + 1e-6).log()
    return point_conf >= threshold
