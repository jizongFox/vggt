import json
import shutil
import tempfile
import typing as t
from pathlib import Path

import numpy as np
import open3d as o3d
import uvicorn
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from servers.inference_interface import run_inference

app = FastAPI(title="VGGT Inference API")


# Custom encoder to handle numpy arrays and Path objects
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, o3d.geometry.PointCloud):
            # Convert Open3D point cloud to dictionary with points and colors
            points = np.asarray(obj.points).tolist() if len(obj.points) > 0 else []
            colors = np.asarray(obj.colors).tolist() if len(obj.colors) > 0 else []
            return {"points": points, "colors": colors}
        return super(NumpyEncoder, self).default(obj)


class InferenceRequest(BaseModel):
    conf_threshold: float = 0.001


@app.post("/inference/")
async def inference(
    background_tasks: BackgroundTasks,
    files: t.List[UploadFile] = File(...),
    conf_threshold: float = 0.001
):
    """
    Endpoint to perform inference using the VGGT model.
    Accepts multiple image files and returns inference results.
    """
    # Create temporary directory to store uploaded files
    temp_dir = Path(tempfile.mkdtemp())
    image_paths = []

    try:
        # Save uploaded files to temporary directory
        for file in files:
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_paths.append(file_path)

        # Run inference
        result = run_inference(image_paths, conf_threshold=conf_threshold)

        # Convert result to serializable format
        serialized_result = json.loads(json.dumps(
            {
                "image_paths": result.image_paths,
                "intrinsic": result.intrinsic,
                "c2w": result.c2w,
                "depths": result.depths,
                "depth_conf": result.depth_conf,
                "points": result.points,
                "point_conf": result.point_conf,
                "colored_pointcloud_from_pointmap": result.colored_pointcloud_from_pointmap,
                "colored_pointcloud_from_depthmap": result.colored_pointcloud_from_depthmap
            },
            cls=NumpyEncoder
        ))

        # Schedule cleanup of temporary files
        background_tasks.add_task(lambda: shutil.rmtree(temp_dir))

        return JSONResponse(content=serialized_result)

    except Exception as e:
        # If an error occurs, ensure temp files are cleaned up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    # Start the server
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
