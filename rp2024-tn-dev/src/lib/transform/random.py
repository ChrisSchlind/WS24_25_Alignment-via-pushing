import numpy as np
from transform.affine import Affine


def sample_point_from_segment(point_a, point_b):
    r = np.random.uniform()
    relative_grasp_point = r * point_a.translation + (1 - r) * point_b.translation
    return Affine(translation=relative_grasp_point)


def sample_pose_from_segment(point_a, point_b):
    # assume some default direction ... e.g. x axis parallel to line, z showing upwards (in planar case)
    # or as up as it can --> what about when x is vertical? solution: perspective is always from above?
    # we assume, that line is horizontal for now ... TODO implement different plugin for different logic
    assert point_a.translation[2] == point_b.translation[2], 'horizontal lines only -> equal z values for points'

    relative_grasp_point = sample_point_from_segment(point_a, point_b)
    if np.random.random() < 0.5:
        x_dir = point_a.translation - point_b.translation
    else:
        x_dir = point_b.translation - point_a.translation

    x_axis = x_dir / np.linalg.norm(x_dir)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)

    rot_m = np.vstack([x_axis, y_axis, z_axis])

    relative_grasp_pose = Affine(translation=relative_grasp_point.translation, rotation=rot_m)

    return relative_grasp_pose


def sample_pose_from_rectangle(point_a, point_b, point_c, point_d):
    # TODO check if points in rectangle
    r_1 = np.random.uniform()
    r_2 = np.random.uniform()
    point_ab = r_1 * point_a.translation + (1 - r_1) * point_b.translation
    point_cd = (1 - r_1) * point_c.translation + r_1 * point_d.translation
    relative_grasp_point = r_2 * point_ab + (1 - r_2) * point_cd

    random_rotation = Affine.random(r_bounds=((0, 0), (0, 0), (0, 2 * np.pi)))

    relative_grasp_pose = Affine(translation=relative_grasp_point, rotation=random_rotation.rotation)

    return relative_grasp_pose
