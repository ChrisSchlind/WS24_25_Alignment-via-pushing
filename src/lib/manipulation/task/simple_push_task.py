import os
from typing import List, Dict, Any, Union
import random
import numpy as np
from dataclasses import dataclass, field
import json

from manipulation.simulation_object import SceneObject, is_overlapping
from transform.affine import Affine


class PushObjectFactory:
    def __init__(self, objects_root: str, object_types: Union[List[str], None] = None):
        self.objects_root = objects_root
        if object_types is None:
            self.object_types = [f.name for f in os.scandir(objects_root) if f.is_dir()]
        else:
            self.object_types = object_types

    def create_push_object(self, object_type: str):
        urdf_path = f'{self.objects_root}/{object_type}/object.urdf'
        kwargs = {'urdf_path': urdf_path}
        push_config_path = f'{self.objects_root}/{object_type}/push_config.json'
        with open(push_config_path) as f:
            push_args = json.load(f)
        kwargs.update(push_args)
        offset = Affine(**push_args['offset']).matrix
        kwargs['offset'] = offset
        return PushObject(**kwargs)


@dataclass
class PushObject(SceneObject):
    """
    Class for objects that can be pushed. A push configuration is required.

    For several objects, there are multiple valid push poses for a successful push execution. In this case we
    restrict ourselves to planar push actions. This reduces the possible push areas to points and segments.
    We have only implemented segments, because a segment with identical endpoints represents a point.
    """
    urdf_path: str
    offset: np.ndarray
    push_poses: List[Dict[str, Any]] = field(default_factory=list)

    def push_to_destination(self, destination: np.ndarray):
        """
        Push the object to the specified destination.
        """
        # Implement the logic to push the object to the destination using pybullet
        pass


class PushTaskFactory:
    def __init__(self, n_objects: int, t_bounds, 
                 r_bounds: np.ndarray = np.array([[0, 0], [0, 0], [0, 2 * np.pi]]),
                 push_object_factory: PushObjectFactory = None):
        self.n_objects = n_objects
        self.t_bounds = t_bounds
        self.r_bounds = r_bounds

        self.unique_id_counter = 0
        self.push_object_factory = push_object_factory

    def get_unique_id(self):
        self.unique_id_counter += 1
        return self.unique_id_counter - 1

    def create_task(self):
        self.unique_id_counter = 0
        n_objects = np.random.randint(1, self.n_objects + 1)
        object_types = random.choices(self.push_object_factory.object_types, k=n_objects)
        push_objects = []
        for object_type in object_types:
            push_object = self.generate_push_object(object_type, push_objects)
            push_objects.append(push_object)

        return PushTask(push_objects)

    def generate_push_object(self, object_type, added_objects):
        manipulation_object = self.push_object_factory.create_push_object(object_type)
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


class PushTask:
    def __init__(self, push_objects: List[PushObject]):
        self.push_objects = push_objects

    def get_info(self):
        info = {
            '_target_': 'manipulation.task.simple_push_task.PushTask',
            'push_objects': self.push_objects,
        }
        return info

    def get_object_with_unique_id(self, unique_id: int):
        for o in self.push_objects:
            if o.unique_id == unique_id:
                return o
        raise RuntimeError('object id mismatch')

    def setup(self, env):
        for o in self.push_objects:
            new_object_id = env.add_object(o)
            o.object_id = new_object_id

    def clean(self, env):
        for o in self.push_objects:
            env.remove_object(o.object_id)


class PushTaskOracle:
    def __init__(self, pusher_offset):
        self.pusher_offset = Affine(**pusher_offset).matrix

    def solve(self, task: PushTask):
        push_object = random.choice(task.push_objects)
        return self.get_push_pose(push_object)
    
    def solve_all(self, task: PushTask):
        solutions = []
        for o in task.push_objects:
            solutions.append(self.get_push_pose(o))
        return solutions
    
    def get_push_pose(self, push_object: PushObject):
        push_pose = random.choice(push_object.push_poses)
        push_pose = push_object.pose @ Affine(**push_pose).matrix @ self.pusher_offset
        return push_pose
