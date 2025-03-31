from pathlib import Path

import open3d as o3d

from client_interface import VGGTClient, ProcessedInferenceResult

# Create client instance
client = VGGTClient(server_url="http://localhost:8000")

# Check if server is running
if client.health_check():
    # Run inference
    image_paths = [
        Path(
            "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012346/0000000107.jpeg"
        ),
        Path(
            "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012347/0000000107.jpeg"
        ),
        Path(
            "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012348/0000000107.jpeg"
        ),
        Path(
            "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN20230102350/0000000107.jpeg"
        ),

        # Path(
        # "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012346/0000000115.jpeg"),
        # Path(
        #     "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012347/0000000115.jpeg"),
        # Path(
        #     "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN2023012348/0000000115.jpeg"),
        # Path(
        #     "/home/jizong/Workspace/dConstruct/data/pddsitpncgaussian/subregion/images/DECXIN20230102350/0000000115.jpeg"),

    ]
    results = client.run_inference(image_paths, conf_threshold=0.01)

    # Process results
    processed_data: ProcessedInferenceResult = client.process_response(results)

    # Access results
    intrinsic = processed_data['intrinsic']
    c2w = processed_data['c2w']
    depths = processed_data['depths']

    # Convert dictionary point cloud data to Open3D PointCloud objects
    pointcloud = processed_data["colored_pointcloud_from_pointmap"]
    pointcloud_from_depth = processed_data["colored_pointcloud_from_depthmap"]

    # Print basic info about the results
    print(f"Processed {len(processed_data['image_paths'])} images")
    print(f"Depth maps shape: {depths.shape}")
    print(f"Point cloud has {len(pointcloud.points)} points")
    print(f"Point cloud from depth has {len(pointcloud_from_depth.points)} points")

    # Visualize the point clouds
    print("Visualizing point cloud from depth map...")
    o3d.visualization.draw_geometries([pointcloud_from_depth])

    print("Visualizing point cloud from point map...")
    o3d.visualization.draw_geometries([pointcloud])
