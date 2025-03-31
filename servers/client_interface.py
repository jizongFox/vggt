import os
import argparse
import requests
import json
from pathlib import Path
import typing as t
from typing import List, Dict, Any, Optional, Union
from typing_extensions import TypedDict
import numpy as np
import open3d as o3d


# TypedDict for processed inference results
class ProcessedInferenceResult(TypedDict, total=False):
    """TypedDict representing the processed inference results."""
    intrinsic: np.ndarray  # shape: batch 3 3
    c2w: np.ndarray  # shape: batch 4 4
    depths: np.ndarray  # shape: batch h w
    depth_conf: np.ndarray  # shape: batch h w
    points: np.ndarray  # shape: batch h w 3
    point_conf: np.ndarray  # shape: batch h w
    colored_pointcloud_from_pointmap: o3d.geometry.PointCloud  # Open3D PointCloud object
    colored_pointcloud_from_depthmap: o3d.geometry.PointCloud  # Open3D PointCloud object
    image_paths: List[str]  # List of image paths as strings


class VGGTClient:
    """
    Client for the VGGT inference API.
    """
    
    def __init__(self, server_url="http://localhost:8000"):
        """
        Initialize the VGGT client.
        
        Args:
            server_url (str): URL of the VGGT inference server
        """
        self.server_url = server_url.rstrip('/')
        
    def health_check(self):
        """Check if the server is running."""
        try:
            response = requests.get(f"{self.server_url}/health")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
            
    def run_inference(self, image_paths, conf_threshold=0.001):
        """
        Run inference on the provided images.
        
        Args:
            image_paths (list): List of paths to images
            conf_threshold (float): Confidence threshold for filtering
            
        Returns:
            dict: Inference results
        """
        if not self.health_check():
            raise ConnectionError(f"Cannot connect to server at {self.server_url}")
        
        # Prepare files for upload
        files = []
        file_objects = []  # Keep track of file objects separately
        for i, path in enumerate(image_paths):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            file_obj = open(path, 'rb')
            file_objects.append(file_obj)
            filename = str(path)

            files.append(
                ('files', (path.parts[-2]+"_"+path.name, file_obj, 'image/jpeg'))
            )
        
        try:
            # Make the request
            response = requests.post(
                f"{self.server_url}/inference/",
                files=files,
                params={"conf_threshold": conf_threshold}
            )
            
            if response.status_code != 200:
                raise Exception(f"Server error: {response.text}")
            
            return response.json()
        finally:
            # Close all opened files
            for file_obj in file_objects:
                file_obj.close()
    
    @staticmethod
    def dict_to_o3d_pointcloud(pointcloud_dict: Dict[str, List[List[float]]]) -> o3d.geometry.PointCloud:
        """
        Convert a dictionary representation of a point cloud to an Open3D PointCloud object.
        
        Args:
            pointcloud_dict: Dictionary with "points" and "colors" keys
            
        Returns:
            o3d.geometry.PointCloud: An Open3D point cloud object
        """
        pcd = o3d.geometry.PointCloud()
        
        if 'points' in pointcloud_dict and pointcloud_dict['points']:
            pcd.points = o3d.utility.Vector3dVector(np.array(pointcloud_dict['points'], dtype=np.float64))
            print(f"Point cloud has {len(pcd.points)} points")
        
        if 'colors' in pointcloud_dict and pointcloud_dict['colors']:
            pcd.colors = o3d.utility.Vector3dVector(np.array(pointcloud_dict['colors'], dtype=np.float64))
            
        return pcd
    
    def process_response(self, response: Dict[str, Any]) -> ProcessedInferenceResult:
        """
        Process and convert response data to more usable formats.
        
        Args:
            response (dict): Server response JSON
            
        Returns:
            ProcessedInferenceResult: Processed data with numpy arrays
        """
        processed: ProcessedInferenceResult = {}
        
        # Convert lists back to numpy arrays where appropriate
        if 'intrinsic' in response:
            processed['intrinsic'] = np.array(response['intrinsic'])
        if 'c2w' in response:
            processed['c2w'] = np.array(response['c2w'])
        if 'depths' in response:
            processed['depths'] = np.array(response['depths'])
        if 'depth_conf' in response:
            processed['depth_conf'] = np.array(response['depth_conf'])
        if 'points' in response:
            processed['points'] = np.array(response['points'])
        if 'point_conf' in response:
            processed['point_conf'] = np.array(response['point_conf'])
        
        # Convert point cloud dictionaries to Open3D PointCloud objects
        if 'colored_pointcloud_from_pointmap' in response:
            processed['colored_pointcloud_from_pointmap'] = self.dict_to_o3d_pointcloud(
                response['colored_pointcloud_from_pointmap']
            )
        
        if 'colored_pointcloud_from_depthmap' in response:
            processed['colored_pointcloud_from_depthmap'] = self.dict_to_o3d_pointcloud(
                response['colored_pointcloud_from_depthmap']
            )
        
        # Keep image paths as strings
        if 'image_paths' in response:
            processed['image_paths'] = response['image_paths']
        
        return processed


def main():
    parser = argparse.ArgumentParser(description="VGGT Inference Client")
    parser.add_argument("--server", type=str, default="http://localhost:8000", 
                        help="URL of the VGGT inference server")
    parser.add_argument("--images", type=str, nargs="+", required=True,
                        help="Paths to image files for inference")
    parser.add_argument("--conf-threshold", type=float, default=0.001,
                        help="Confidence threshold for filtering")
    parser.add_argument("--output", type=str, default="inference_results.json",
                        help="Output file to save results")
    
    args = parser.parse_args()
    
    # Create client
    client = VGGTClient(args.server)
    
    # Check server connection
    if not client.health_check():
        print(f"Error: Cannot connect to server at {args.server}")
        return
    
    print(f"Connected to server at {args.server}")
    print(f"Running inference on {len(args.images)} images...")
    
    # Run inference
    try:
        response = client.run_inference(args.images, args.conf_threshold)
        processed = client.process_response(response)
        
        # Save results to file
        with open(args.output, 'w') as f:
            json.dump(response, f, indent=2)
        
        print(f"Inference completed successfully!")
        print(f"Results saved to {args.output}")
        
        # Print some basic info
        print("\nInference Summary:")
        print(f"Number of images: {len(processed['image_paths'])}")
        if 'depths' in processed:
            print(f"Depth map shape: {processed['depths'].shape}")
        if 'points' in processed:
            print(f"Point map shape: {processed['points'].shape}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")


if __name__ == "__main__":
    main()
