import numpy as np
import pybullet as p
from transform.affine import Affine
from bullet_env.camera import BulletCamera


class ParallaxCamera(BulletCamera):
    def __init__(self, bullet_client, t_center, height, resolution, intrinsics, depth_range, record_depth):
        # Adjust the rotation to look down at the table
        rotation = Affine(rotation=[0, 1, 0, 0]).matrix[:3, :3]  # Correct rotation matrix        
        pose_matrix = Affine(translation=[t_center[0], t_center[1], height], rotation=rotation).matrix
        super().__init__(bullet_client, pose_matrix, resolution, intrinsics, depth_range, record_depth)
