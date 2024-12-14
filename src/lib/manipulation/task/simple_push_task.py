import os
from typing import List, Dict, Any, Union
import random
import numpy as np
from dataclasses import dataclass, field
import json

from manipulation.simulation_object import SceneObject, is_overlapping
from transform.affine import Affine
from transform.random import sample_pose_from_segment


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

    For several objects, there are multiple valid pusher poses for a successful push execution. In this case we
    restrict ourselves to planar push actions with a 2-jaw parallel pusher. This reduces the possible push areas
    to points and segments. We have only implemented segments, because a segment with identical endpoints
    represents a point.
    """
    static: bool = False
    push_config: List[Dict[str, Any]] = field(default_factory=lambda: [])

    def get_valid_pose(self):
        """
        This method samples and returns a valid pusher pose relative to the object's pose, based on the segments
        defined in the push configuration.
        """
        push_area = random.sample(self.push_config, 1)[0]

        valid_pose = None

        if push_area['type'] == 'segment':
            point_a = Affine(translation=push_area['point_a'])
            point_b = Affine(translation=push_area['point_b'])
            valid_pose = sample_pose_from_segment(point_a, point_b)

        return valid_pose


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

    def setup(self, env):
        for obj in self.push_objects:
            env.add_object(obj)

    def get_info(self):
        return {"push_objects": [obj.get_info() for obj in self.push_objects]}

    def clean(self, env):
        for obj in self.push_objects:
            env.remove_object(obj)