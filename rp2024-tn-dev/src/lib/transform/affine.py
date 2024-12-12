import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy import interpolate


class Affine:
    """
    Class for representing and dealing with affine transformations.
    """

    def __init__(self, translation=(0, 0, 0), rotation=(0, 0, 0, 1)):
        self.matrix = np.eye(4)
        self.matrix[:3, 3] = np.array(translation)
        if len(rotation) == 4:
            rot_matrix = Rotation.from_quat(rotation).as_matrix()
        elif len(rotation) == 3:
            rot_matrix = Rotation.from_euler('xyz', rotation).as_matrix()
            if len(np.array(rotation).shape) > 1:
                rot_matrix = np.array(rotation)
        else:
            raise ValueError('Expected `rotation` to have shape (4,) or (3,), got ' + str(np.array(rotation).shape))

        self.matrix[:3, :3] = rot_matrix

    @classmethod
    def from_matrix(cls, matrix):
        affine = cls()
        affine.matrix = matrix
        return affine

    @classmethod
    def random(cls,
               t_bounds=((0, 1), (0, 1), (0, 1)),
               r_bounds=((0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)),
               allow_zero_translation=True,
               allow_zero_rotation=True):
        # TODO: proper rotation randomization in 3D with axis angle representation
        t_b = np.array(t_bounds)
        translation = np.array([0.0, 0.0, 0.0])
        t_norm = 0.0
        if not allow_zero_translation:
            while t_norm < 0.0001:
                translation = np.random.uniform(t_b[:, 0], t_b[:, 1])
                t_norm = np.linalg.norm(translation)
        else:
            translation = np.random.uniform(t_b[:, 0], t_b[:, 1])
        r_b = np.array(r_bounds)
        rpy = np.array([0.0, 0.0, 0.0])
        if not allow_zero_rotation:
            while (rpy < 0.0001).all():
                rpy = np.random.uniform(r_b[:, 0], r_b[:, 1])
        else:
            rpy = np.random.uniform(r_b[:, 0], r_b[:, 1])
        rotation = Rotation.from_euler('xyz', rpy).as_quat()
        return cls(translation=translation, rotation=rotation)

    @classmethod
    def polar(cls, azimuth, polar, radius, t_center):
        x = radius * np.sin(polar) * np.cos(azimuth)
        y = radius * np.sin(polar) * np.sin(azimuth)
        z = radius * np.cos(polar)
        t = np.array([x, y, z])
        t += t_center
        # compute rotation matrix, camera looks at center
        z_axis = t_center - t
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        if np.linalg.norm(x_axis) == 0:
            x_axis = np.array([np.cos(azimuth), np.sin(azimuth), 0])
        else:
            x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        r = np.stack([x_axis, y_axis, z_axis], axis=1)
        return cls(translation=t, rotation=r)

    '''
     Special methods for representing the translation and rotations
    '''

    def __repr__(self):
        return str(self.translation) + ' ' + str(self.quat)

    def __str__(self):
        return str(self.translation) + ' ' + str(self.quat)

    def __mul__(self, other):
        return Affine.from_matrix(self.matrix @ other.matrix)

    def __truediv__(self, other):
        return other.invert() * self

    @property
    def rotation(self):
        """
        setter method for giving the rotation
        """
        return self.matrix[:3, :3]

    @property
    def translation(self):
        """
        setter method for giving the translation
        """
        return self.matrix[:3, 3]

    @property
    def quat(self):
        """
        setter method for returning the quaternions
        """
        return Rotation.from_matrix(self.matrix[:3, :3]).as_quat()

    @property
    def rpy(self):
        """
        setter method for returning the euler
        """
        return Rotation.from_matrix(self.matrix[:3, :3]).as_euler('xyz')

    @property
    def rotvec(self):
        """
        setter method for returning the rotation vector
        """
        return Rotation.from_matrix(self.matrix[:3, :3]).as_rotvec()

    def invert(self):
        """
        This method is going the inverse the rotational matrix.
        """
        return Affine.from_matrix(np.linalg.inv(self.matrix))

    def to_twist(self):
        R = self.matrix[:3, :3]
        t = self.matrix[:3, 3]
        theta = np.arccos((np.trace(R) - 1) / 2)
        if theta != 0:
            omega_hat = 1 / (2 * np.sin(theta)) * (R - R.T)
            omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])
            omega = omega * theta
            V_inv_theta = np.eye(3) / theta - 0.5 * omega_hat + (1 / theta - 1 / (2 * np.tan(theta / 2))) * np.matmul(
                omega_hat,
                omega_hat)
            v = np.matmul(V_inv_theta, t.reshape((3, 1)))
        else:
            omega = np.zeros((3,))
            v = t
        twist = np.concatenate([omega, v.reshape((3,))], axis=0)
        return twist

    def interpolate_to(self, transform, lin_step_size):
        t_start = self.matrix[:3, 3]
        t_goal = transform.matrix[:3, 3]
        # TODO: only works if translations are far enough. Fix for rotations too!
        l = np.linalg.norm(t_goal - t_start)
        if l < 2 * lin_step_size:
            return [self, transform]
        n_steps = int(np.linalg.norm(t_goal - t_start) / lin_step_size)
        key_steps = np.arange(n_steps)
        interp = interpolate.interp1d([0, n_steps - 1], [t_start, t_goal], axis=0)
        t_steps = interp(key_steps)
        rs = Rotation.from_matrix([self.matrix[:3, :3], transform.matrix[:3, :3]])
        slerp = Slerp([0, n_steps - 1], rs)
        r_steps = slerp(key_steps)
        steps = [Affine(t, r.as_quat()) for t, r in zip(t_steps, r_steps)]
        return steps

