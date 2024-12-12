import os
from typing import List, Dict, Any, Union
import random
import numpy as np
from dataclasses import dataclass, field
import json

from manipulation.simulation_object import SceneObject, is_overlapping
from transform.affine import Affine
from transform.random import sample_pose_from_segment


class GraspObjectFactory:
    def __init__(self, objects_root: str, object_types: Union[List[str], None] = None):
        self.objects_root = objects_root
        if object_types is None:
            self.object_types = [f.name for f in os.scandir(objects_root) if f.is_dir()]
        else:
            self.object_types = object_types

    def create_grasp_object(self, object_type: str):
        urdf_path = f'{self.objects_root}/{object_type}/object.urdf'
        kwargs = {'urdf_path': urdf_path}
        grasp_config_path = f'{self.objects_root}/{object_type}/grasp_config.json'
        with open(grasp_config_path) as f:
            grasp_args = json.load(f)
        kwargs.update(grasp_args)
        offset = Affine(**grasp_args['offset']).matrix
        kwargs['offset'] = offset
        return GraspObject(**kwargs)


@dataclass
class GraspObject(SceneObject):
    """
    Class for objects that can be picked. A pick configuration is required.

    For several objects, there are multiple valid gripper poses for a successful pick execution. In this case we
    restrict ourselves to planar pick actions with a 2-jaw parallel gripper. This reduces the possible pick areas
    to points and segments. We have only implemented segments, because a segment with identical endpoints
    represents a point.
    """
    static: bool = False
    grasp_config: List[Dict[str, Any]] = field(default_factory=lambda: [])

    def get_valid_pose(self):
        """
        This method samples and returns a valid gripper pose relative to the object's pose, based on the segments
        defined in the pick configuration.
        """
        grasp_area = random.sample(self.grasp_config, 1)[0]

        valid_pose = None

        if grasp_area['type'] == 'segment':
            point_a = Affine(translation=grasp_area['point_a'])
            point_b = Affine(translation=grasp_area['point_b'])
            valid_pose = sample_pose_from_segment(point_a, point_b)

        if valid_pose is None:
            raise Exception(f'No valid pose found for pick object {self}')
        return valid_pose.matrix


class GraspTaskFactory:
    def __init__(self, n_objects: int, t_bounds, 
                 r_bounds: np.ndarray = np.array([[0, 0], [0, 0], [0, 2 * np.pi]]),
                 grasp_object_factory: GraspObjectFactory = None):
        self.n_objects = n_objects
        self.t_bounds = t_bounds
        self.r_bounds = r_bounds

        self.unique_id_counter = 0
        self.grasp_object_factory = grasp_object_factory

    def get_unique_id(self):
        self.unique_id_counter += 1
        return self.unique_id_counter - 1

    def create_task(self):
        self.unique_id_counter = 0
        n_objects = np.random.randint(1, self.n_objects + 1)
        object_types = random.choices(self.grasp_object_factory.object_types, k=n_objects)
        grasp_objects = []
        for object_type in object_types:
            grasp_object = self.generate_grasp_object(object_type, grasp_objects)
            grasp_objects.append(grasp_object)

        return GraspTask(grasp_objects)

    def generate_grasp_object(self, object_type, added_objects):
        manipulation_object = self.grasp_object_factory.create_grasp_object(object_type)
        object_pose = self.get_non_overlapping_pose(manipulation_object.min_dist, added_objects)
        corrected_pose = manipulation_object.offset @ object_pose.matrix
        manipulation_object.pose = corrected_pose
        manipulation_object.unique_id = self.get_unique_id()
        return manipulation_object

    def get_non_overlapping_pose(self, min_dist, objects):
        overlapping = True
        new_t_bounds = np.array(self.t_bounds)
        new_t_bounds[:2, 0] = new_t_bounds[:2, 0] + min_dist
        new_t_bounds[:2, 1] = new_t_bounds[:2, 1] - min_dist
        while overlapping:
            random_pose = Affine.random(t_bounds=new_t_bounds, r_bounds=self.r_bounds)
            overlapping = is_overlapping(random_pose, min_dist, objects)
        return random_pose


class GraspTask:
    def __init__(self, grasp_objects: List[GraspObject]):
        self.grasp_objects = grasp_objects

    def get_info(self):
        info = {
            '_target_': 'manipulation.task.simple_grasp_task.GraspTask',
            'grasp_objects': self.grasp_objects,
        }
        return info

    def get_object_with_unique_id(self, unique_id: int):
        for o in self.grasp_objects:
            if o.unique_id == unique_id:
                return o
        raise RuntimeError('object id mismatch')

    def setup(self, env):
        for o in self.grasp_objects:
            new_object_id = env.add_object(o)
            o.object_id = new_object_id

    def clean(self, env):
        for o in self.grasp_objects:
            env.remove_object(o.object_id)


class GraspTaskOracle:
    def __init__(self, gripper_offset):
        self.gripper_offset = Affine(**gripper_offset).matrix

    def solve(self, task: GraspTask):
        manipulation_object = random.choice(task.grasp_objects)
        return self.get_grasp_pose(manipulation_object)
    
    def solve_all(self, task: GraspTask):
        solutions = []
        for o in task.grasp_objects:
            solutions.append(self.get_grasp_pose(o))
        return solutions
    
    def get_grasp_pose(self, manipulation_object: GraspObject):
        pick_pose = manipulation_object.get_valid_pose()
        pick_pose = manipulation_object.pose @ pick_pose @ self.gripper_offset
        return pick_pose
    