import numpy as np
import pybullet as p
from transform.affine import Affine
from bullet_env.camera import BulletCamera


class TeletentricCamera(BulletCamera):
    def __init__(self, bullet_client, t_center, height, resolution, intrinsics, depth_range, record_depth):
        # Adjust the rotation to look down at the table
        rotation = Affine(rotation=[0, 1, 0, 0]).matrix[:3, :3]  # Correct rotation matrix
        pose_matrix = Affine(translation=[t_center[0], t_center[1], height], rotation=rotation).matrix
        super().__init__(bullet_client, pose_matrix, resolution, intrinsics, depth_range, record_depth)

    def get_observation(self):
        """Get RGB and depth observation from the camera."""
        _, _, color, depth, _ = self.bullet_client.getCameraImage(
            width=self.resolution[0],
            height=self.resolution[1],
            viewMatrix=self.view_m,
            projectionMatrix=self.proj_m,
            shadow=1,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        color = np.array(color).reshape(self.resolution[1], self.resolution[0], -1)[..., :3].astype(np.uint8)
        observation = {'rgb': color,
                       'extrinsics': self.pose.matrix,
                       'intrinsics': np.reshape(self.intrinsics, (3, 3)).astype(np.float32)}
        if self.record_depth:
            depth_buffer_opengl = np.reshape(depth, [self.resolution[1], self.resolution[0]])
            depth_opengl = self.depth_range[1] * self.depth_range[0] / (self.depth_range[1] - (self.depth_range[1] - self.depth_range[0]) * depth_buffer_opengl)
            observation['depth'] = depth_opengl
        return observation
