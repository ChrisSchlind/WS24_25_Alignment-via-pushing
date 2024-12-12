import numpy as np
import pybullet as p

from transform.affine import Affine
import numpy as np


def cvK2BulletP(K, w, h, near, far):
    """
    cvKtoPulletP converst the K interinsic matrix as calibrated using Opencv
    and ROS to the projection matrix used in openGL and Pybullet.

    :param K:  OpenCV 3x3 camera intrinsic matrix
    :param w:  Image width
    :param h:  Image height
    :near:     The nearest objects to be included in the render
    :far:      The furthest objects to be included in the render
    :return:   4x4 projection matrix as used in openGL and pybullet
    """ 
    f_x = K[0,0]
    f_y = K[1,1]
    c_x = K[0,2]
    c_y = K[1,2]
    A = (near + far)/(near - far)
    B = 2 * near * far / (near - far)

    projection_matrix = [
                        [2/w * f_x,  0,          (w - 2*c_x)/w,  0],
                        [0,          2/h * f_y,  (2*c_y - h)/h,  0],
                        [0,          0,          A,              B],
                        [0,          0,          -1,             0]]
    #The transpose is needed for respecting the array structure of the OpenGL
    return np.array(projection_matrix).T.reshape(16).tolist()


def cvPose2BulletView(q, t):
    """
    cvPose2BulletView gets orientation and position as used 
    in ROS-TF and opencv and coverts it to the view matrix used 
    in openGL and pyBullet.
    
    :param q: ROS orientation expressed as quaternion [qx, qy, qz, qw] 
    :param t: ROS postion expressed as [tx, ty, tz]
    :return:  4x4 view matrix as used in pybullet and openGL
    
    """
    q = Quaternion([q[3], q[0], q[1], q[2]])
    R = q.rotation_matrix

    T = np.vstack([np.hstack([R, np.array(t).reshape(3,1)]),
                                np.array([0, 0, 0, 1])])
    # Convert opencv convention to python convention
    # By a 180 degrees rotation along X
    Tc = np.array([[1,   0,    0,  0],
                    [0,  -1,    0,  0],
                    [0,   0,   -1,  0],
                    [0,   0,    0,  1]]).reshape(4,4)
    
    # pybullet pse is the inverse of the pose from the ROS-TF
    T=Tc@np.linalg.inv(T)
    # The transpose is needed for respecting the array structure of the OpenGL
    viewMatrix = T.T.reshape(16)
    return viewMatrix


class BulletCamera:
    def __init__(self, bullet_client, pose_matrix, resolution=(640, 480),
                 intrinsics=(450, 0, 320, 0, 450, 240, 0, 0, 1),
                 depth_range=(0.01, 10.0),
                 record_depth=False):
        self.bullet_client = bullet_client
        self.pose = Affine.from_matrix(pose_matrix)
        self.resolution = resolution
        self.intrinsics = intrinsics
        self.depth_range = depth_range
        self.record_depth = record_depth
        self.view_m = self.compute_view_matrix()
        # self.proj_m = self.compute_projection_matrix()
        self.proj_m = cvK2BulletP(np.reshape(self.intrinsics, (3, 3)), self.resolution[0], self.resolution[1], *self.depth_range)

    def compute_projection_matrix(self):
        focal_len = self.intrinsics[0]
        z_near, z_far = self.depth_range
        half_fov = (self.resolution[1] / 2) / focal_len
        fov = 2 * np.arctan(half_fov) * 180 / np.pi
        aspect_ratio = self.resolution[0] / self.resolution[1]
        proj_m = self.bullet_client.computeProjectionMatrixFOV(fov, aspect_ratio, z_near, z_far)
        return proj_m

    def compute_view_matrix(self):
        front_dir = Affine(translation=[0, 0, 1])
        up_dir = Affine(translation=[0, -1, 0])
        front_dir = self.pose * front_dir
        up_dir = self.pose.rotation @ up_dir.translation
        view_m = self.bullet_client.computeViewMatrix(self.pose.translation, front_dir.translation, up_dir)
        return view_m

    def get_observation(self):
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


class CameraFactory:
    @property
    def cameras(self):
        raise NotImplementedError


class StaticCameraFactory(CameraFactory):
    def __init__(self, bullet_client, camera_configs):
        self.bullet_client = bullet_client
        self._cameras = [BulletCamera(bullet_client, **config) for config in camera_configs]

    @property
    def cameras(self):
        return self._cameras


class StaticPolarCameraFactory(CameraFactory):
    def __init__(self, bullet_client, camera_config, radius, t_center, polar_angles, upside_down=False):
        self.bullet_client = bullet_client
        poses = [Affine.polar(pose[0], pose[1], radius, t_center) for pose in polar_angles]
        if upside_down:
            poses = [pose * Affine(rotation=[0, 0, np.pi]) for pose in poses]
        self._cameras = [BulletCamera(bullet_client, pose.matrix, **camera_config) for pose in poses]

    @property
    def cameras(self):
        return self._cameras


class PolarCameraFactory(CameraFactory):
    def __init__(self, bullet_client,
                 t_center,
                 camera_config,
                 n_perspectives,
                 min_azimuth=-np.pi + np.pi / 6,
                 max_azimuth=np.pi - np.pi / 6,
                 min_polar=np.pi / 6,
                 max_polar=np.pi / 3,
                 radius=0.8,
                 upside_down=False):
        self.bullet_client = bullet_client
        self.camera_config = camera_config
        self.t_center = t_center
        self.n_perspectives = n_perspectives
        self.min_azimuth = min_azimuth
        self.max_azimuth = max_azimuth
        self.min_polar = min_polar
        self.max_polar = max_polar
        self.radius = radius
        self.upside_down = upside_down

    def create_camera(self):
        azimuth = np.random.uniform(self.min_azimuth, self.max_azimuth)
        cos_polar = np.random.uniform(np.cos(self.max_polar), np.cos(self.min_polar))
        polar = np.arccos(cos_polar)
        pose = Affine.polar(azimuth, polar, self.radius, self.t_center)
        if self.upside_down:
            pose = pose * Affine(rotation=[0, 0, np.pi])
        return BulletCamera(self.bullet_client, pose.matrix, **self.camera_config)

    @property
    def cameras(self):
        cameras = [self.create_camera() for _ in range(self.n_perspectives)]
        return cameras
