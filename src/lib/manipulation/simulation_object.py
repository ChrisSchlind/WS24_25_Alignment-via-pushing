from dataclasses import dataclass, field
import numpy as np
from transform.affine import Affine


def is_overlapping(pose, min_dist, objects):
    for o in objects:
        o_pose = Affine.from_matrix(o.pose)
        d = np.linalg.norm(pose.translation[:2] - o_pose.translation[:2])
        overlap = d < (min_dist + o.min_dist)
        if overlap:
            return True
    return False


@dataclass
class SceneObject:
    """
    Base class for objects that can be placed in a scene.

    Class variables:
    :var str urdf_path: path to the urdf describing the physical properties of the object
    :var int object_id: id of object from the simulation - if there is one
    :var bool static: indicates whether the object can be moved
    :var Affine pose: 6D pose of the object
    :var float min_dist: encompassing radius, for non-overlapping object placement
    :var Affine offset: offset of object origin and its base, to avoid placing object into the ground
    :var int unique_id: unique id of object that was generated while task generation. It is used in objectives.
    """
    urdf_path: str = None
    object_id: int = -1
    static: bool = True
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    min_dist: float = 0
    offset: np.ndarray = field(default_factory=lambda: np.eye(4))
    unique_id: int = -1
