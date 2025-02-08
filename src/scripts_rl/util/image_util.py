import cv2
import numpy as np
from transform.affine import Affine


def draw_pose(extrinsics, pose, intrinsics, rgb, length=0.1, thickness=6):
    camera_pose = Affine.from_matrix(extrinsics)
    pose_affine = Affine.from_matrix(pose)
    input_relative_camera_pose = camera_pose.invert() * pose_affine
    o_t = input_relative_camera_pose.translation[..., np.newaxis]
    o_r = input_relative_camera_pose.rotvec[..., np.newaxis]
    cv2.drawFrameAxes(rgb, intrinsics, np.zeros((5, 1)), o_r, o_t, length, thickness)