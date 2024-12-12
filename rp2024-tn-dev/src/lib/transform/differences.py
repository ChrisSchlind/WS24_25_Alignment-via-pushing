import numpy as np
from transform.affine import Affine


def rotation_to_line_difference(rotation: np.array, line_point_a, line_point_b):
    # Assume we search for the x-axis' rotational error to the line
    x_axis = Affine(rotation=rotation) * Affine(translation=(1, 0, 0))
    direction = line_point_b - line_point_a
    direction = direction / np.linalg.norm(direction)

    cos = np.dot(x_axis.translation, direction)
    cos = min(1.0, max(-1.0, float(cos)))
    r_error = np.arccos(np.abs(cos))
    return r_error, cos


def point_to_segment_distance(point: np.array, line_point_a, line_point_b):
    a_b = line_point_b - line_point_a
    a_b_n = a_b / np.linalg.norm(a_b)
    b_p_n = (point - line_point_b) / np.linalg.norm(point - line_point_b)
    cos_a_b_b_p = np.dot(a_b_n, b_p_n)
    a_p = point - line_point_a
    a_p_n = (point - line_point_a) / np.linalg.norm(point - line_point_a)
    cos_a_b_a_p = np.dot(a_b_n, a_p_n)

    if cos_a_b_b_p > 0:
        distance = np.linalg.norm(point - line_point_b)
    elif cos_a_b_a_p < 0:
        distance = np.linalg.norm(point - line_point_a)
    else:
        cross = np.cross(a_b, a_p)
        distance = np.linalg.norm(cross) / np.linalg.norm(a_b)
    return distance


def transformation_difference(pose_a, pose_b):
    translation_error = np.linalg.norm(pose_a.translation - pose_b.translation)
    rotation_error = np.linalg.norm((pose_a.invert() * pose_b).rotvec)
    return translation_error, rotation_error
