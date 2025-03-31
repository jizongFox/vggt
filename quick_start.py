import warnings
from pathlib import Path

import open3d as o3d
import torch
from jaxtyping import Float, Bool
from torch import Tensor

from quick_start_helper import create_colored_pcd, added_camera_center
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT()
_URL = "/home/jizong/.cache/huggingface/hub/model.pt"
model.load_state_dict(torch.load(_URL, map_location=device, weights_only=True))
model.to(device)

# Load and preprocess example images (replace with your own image paths)
# image_names = sorted(Path("/home/jizong/Workspace/dConstruct/vggt/examples/room/images").rglob("*jpg"))[:5]
root_dir = Path("/home/jizong/Workspace/dConstruct/data/20250325-debug/subregion/images")
time_steps = sorted(set([x.stem for x in root_dir.rglob("*.jpeg")]))[::100][2:3]
# breakpoint()
# time_steps = ["0000000968", ""]
cameras = ["DECXIN2023012346", "DECXIN2023012347", "DECXIN2023012348", "DECXIN20230102350"]
# cameras = ["DECXIN2023012346", ]
image_names = []
for time_step in time_steps:
    image_names.extend([root_dir / x / f"{time_step}.jpeg" for x in cameras])
print(time_steps)
images = load_and_preprocess_images(image_names).to(device)


# with torch.no_grad() and torch.cuda.amp.autocast(dtype=dtype), warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=FutureWarning)
#
#     # Predict attributes including cameras, depth maps, and point maps.
#     predictions = model(images)

def filter_by_conf(point_conf: Float[Tensor, "*d 3"],
                   threshold: float = 0.5) -> Bool[Tensor, "*d 3"]:
    point_conf = (point_conf[...] + 1e-6).log()
    return point_conf >= threshold


with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
    images = images[None]  # add batch dimension
    aggregated_tokens_list, ps_idx = model.aggregator(images)

    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    print(f"intrinsic: {intrinsic[0].squeeze()}")

    # maybe extract the camera pose
    extrinsic2 = torch.zeros(extrinsic.shape[1], 4, 4)
    extrinsic2[:, 3, 3] = 1.0
    extrinsic2[:, :3, :] = extrinsic[0]
    extrinsic2 = extrinsic2.inverse()

    torch.save(extrinsic2, "/home/jizong/Workspace/dConstruct/data/20250325-debug/c2w.pt")
    torch.save(intrinsic, "/home/jizong/Workspace/dConstruct/data/20250325-debug/intrinsic.pt")

    camera_centers = extrinsic2[:, :3, 3]

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    torch.save(depth_map, "/home/jizong/Workspace/dConstruct/data/20250325-debug/depth.pt")

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

    # visualize
    pcd = create_colored_pcd(xyz=point_map[0], pcd_xyz_dim=-1, color=images[0], color_dim=1)
    pcd = pcd.voxel_down_sample(voxel_size=0.001)

    pcd = added_camera_center(pcd, camera_centers)

    o3d.visualization.draw_geometries([pcd])

    # visualize filtered
    filtered_mask = filter_by_conf(point_conf, threshold=0.01)

    pcd2 = create_colored_pcd(xyz=point_map[filtered_mask], pcd_xyz_dim=-1, color=images.moveaxis(2, -1)[filtered_mask],
                              color_dim=-1)
    pcd2 = pcd2.voxel_down_sample(voxel_size=0.001)

    pcd2 = added_camera_center(pcd2, camera_centers)

    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(Path("/home/jizong/Workspace/dConstruct/data/20250325-debug/filtered_from_point_map.ply"),
                             pcd2,
                             )

    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0),
                                                                 extrinsic.squeeze(0),
                                                                 intrinsic.squeeze(0))

    filtered_mask = filter_by_conf(depth_conf, threshold=0.01)

    pcd2 = create_colored_pcd(xyz=torch.from_numpy(point_map_by_unprojection[None, ...]).cuda()[filtered_mask],
                              pcd_xyz_dim=-1, color=images.moveaxis(2, -1)[filtered_mask],
                              color_dim=-1)
    # o3d.visualization.draw_geometries([pcd2])
    o3d.io.write_point_cloud(
        Path("/home/jizong/Workspace/dConstruct/data/20250325-debug/filtered_from_point_depth.ply"),
        pcd2,
    )

    # Predict Tracks
    # choose your own points to track, with shape (N, 2) for one scene
    query_points = torch.FloatTensor([[100.0, 200.0],
                                      [60.72, 259.94]]).to(device)

    track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx,
                                                         query_points=query_points[None])

from pointcloud_dc.reconstruct.utils.parser import PCDParsedOutput

pcd_output = PCDParsedOutput.from_export(
    Path("/home/jizong/Workspace/dConstruct/data/20250325-debug/outputs/01bbdb0/exports/trajectory.json"))
