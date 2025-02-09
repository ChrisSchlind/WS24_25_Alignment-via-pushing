import copy
import cv2
import numpy as np
from .convert_util import convert_to_orthographic, display_orthographic

def draw_camera_direction(bullet_client, camera_pose, length=0.1):
    """Draws a line indicating the camera's direction."""
    start_pos = camera_pose.translation
    end_pos = start_pos + camera_pose.rotation @ np.array([0, 0, length])
    bullet_client.addUserDebugLine(start_pos, end_pos, [1, 0, 0], 2)


def update_observation_window(camera_factory, teletentric_camera, cfg):
    observations = [camera.get_observation() for camera in camera_factory.cameras]

    # 1. Standard cameras
    image_copy = copy.deepcopy(observations[0]["rgb"])
    # Convert to rgb for visualization
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    cv2.imshow("Normal Camera No. 0: RGB", image_copy)
    depth_copy = copy.deepcopy(observations[0]["depth"])
    # rescale for visualization
    depth_copy = depth_copy / 2.0
    cv2.imshow("Normal Camera No. 0: Depth", depth_copy)

    # Convert observations to orthographic view
    height_map, colormap = convert_to_orthographic(observations, cfg.workspace_bounds, cfg.projection_resolution)
    # Display orthographic view
    display_orthographic(height_map, colormap, cfg.workspace_bounds)

    # 2. Teletentric camera
    teletentric_observation, _ = teletentric_camera.get_observation()

    teletentric_image_copy = copy.deepcopy(teletentric_observation["rgb"])
    # Convert to rgb for visualization
    teletentric_image_copy = cv2.cvtColor(teletentric_image_copy, cv2.COLOR_BGR2RGB)
    cv2.imshow("Teletentric Camera: RGB", teletentric_image_copy)
    teletentric_depth_copy = copy.deepcopy(teletentric_observation["depth"])
    # rescale for visualization
    teletentric_depth_copy = teletentric_depth_copy / 2.0
    cv2.imshow("Teletentric Camera: Depth", teletentric_depth_copy)