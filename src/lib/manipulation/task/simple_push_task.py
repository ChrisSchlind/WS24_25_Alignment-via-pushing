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
        # Assign a random RGBA color
        rgba_color = [random.random() for _ in range(3)] + [1.0]  # Random RGB and full opacity
        kwargs['color'] = rgba_color
        return PushObject(**kwargs)


@dataclass
class PushObject(SceneObject):
    """
    Class for objects that can be pushed. A push configuration is required.

    """
    static: bool = False
    push_config: List[Dict[str, Any]] = field(default_factory=lambda: [])
    color: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 1.0])

    '''def get_valid_pose(self):
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

        return valid_pose'''
    
class PushAreaFactory:
    def __init__(self, areas_root: str, areas_types: Union[List[str], None] = None):
        self.areas_root = areas_root
        if areas_types is None:
            self.areas_types = [f.name for f in os.scandir(areas_root) if f.is_dir()]
        else:
            self.areas_types = areas_types

    def create_push_area(self, area_type: str, color: List[float]):
        urdf_path = f'{self.areas_root}/{area_type}/area.urdf'
        kwargs = {'urdf_path': urdf_path}
        push_config_path = f'{self.areas_root}/{area_type}/push_config.json'
        with open(push_config_path) as f:
            push_args = json.load(f)
        kwargs.update(push_args)
        offset = Affine(**push_args['offset']).matrix
        kwargs['offset'] = offset
        kwargs['color'] = color
        return PushArea(**kwargs)

    
@dataclass
class PushArea(SceneObject):
    """
    Class for areas that correspond to push objects. The area will have the same color as the push object.
    """

    static: bool = False
    push_config: List[Dict[str, Any]] = field(default_factory=lambda: [])
    color: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 1.0])


class PushTaskFactory:
    def __init__(self, n_objects: int, t_bounds, 
                 r_bounds: np.ndarray = np.array([[0, 0], [0, 0], [0, 2 * np.pi]]),
                 push_object_factory: PushObjectFactory = None,
                 push_area_factory: PushAreaFactory = None,
                 areas_root: str = None):
        self.n_objects = n_objects
        self.t_bounds = t_bounds
        self.r_bounds = r_bounds

        self.unique_id_counter = 0
        self.push_object_factory = push_object_factory
        self.push_area_factory = push_area_factory
        self.areas_root = areas_root

        self.type = 'object'

    def get_unique_id(self):
        self.unique_id_counter += 1
        return self.unique_id_counter - 1

    def create_task(self):
        self.unique_id_counter = 0
        n_objects = np.random.randint(1, self.n_objects + 1)
        object_types = random.choices(self.push_object_factory.object_types, k=n_objects)
        push_objects = []
        push_areas = []
        for object_type in object_types:
            push_object = self.generate_push_object(object_type, push_objects)
            push_objects.append(push_object)
            push_area = self.generate_push_area(object_type, push_objects, push_areas, push_object.color)
            push_areas.append(push_area)

        return PushTask(push_objects, push_areas)

    def generate_push_object(self, object_type, added_objects):
        manipulation_object = self.push_object_factory.create_push_object(object_type)
        self.type = 'object'
        object_pose = self.get_non_overlapping_pose(manipulation_object.min_dist, added_objects)
        corrected_pose = manipulation_object.offset @ object_pose.matrix
        manipulation_object.pose = corrected_pose
        manipulation_object.unique_id = self.get_unique_id()
        return manipulation_object
    
    def generate_push_area(self, area_type, added_objects, added_areas, color):
        manipulation_area = self.push_area_factory.create_push_area(area_type, color)
        self.type = 'area'
        area_pose = self.get_non_overlapping_pose(manipulation_area.min_dist, added_objects, added_areas)
        corrected_pose = manipulation_area.offset @ area_pose.matrix
        manipulation_area.pose = corrected_pose
        manipulation_area.unique_id = self.get_unique_id()
        return manipulation_area

    def get_non_overlapping_pose(self, min_dist, objects, areas=None):
        overlapping = True
        new_t_bounds = np.array(self.t_bounds)
        new_t_bounds[:2, 0] = new_t_bounds[:2, 0] + min_dist
        new_t_bounds[:2, 1] = new_t_bounds[:2, 1] - min_dist
        while overlapping:
            random_pose = Affine.random(t_bounds=new_t_bounds, r_bounds=self.r_bounds)
            overlapping = is_overlapping(random_pose, min_dist, objects)

            if self.type == 'area' and areas is not None: # check overlapping for areas and objects if the type is area
                overlapping = is_overlapping(random_pose, min_dist, areas) or overlapping
        return random_pose
    
    def get_random_pose(self):
        return Affine.random(t_bounds=self.t_bounds, r_bounds=self.r_bounds)


class PushTask:
    def __init__(self, push_objects: List[PushObject], push_areas: List[PushArea]):
        self.push_objects = push_objects
        self.push_areas = push_areas

    def get_info(self):
        info = {
            '_target_': 'manipulation.task.simple_push_task.PushTask',
            'push_objects': self.push_objects,
            'push_areas': self.push_areas
        }
        return info

    def get_object_with_unique_id(self, unique_id: int):
        for o in self.push_objects:
            if o.unique_id == unique_id:
                return o
        raise RuntimeError('object id mismatch')
    
    def get_area_with_unique_id(self, unique_id: int):
        for a in self.push_areas:
            if a.unique_id == unique_id:
                return a
        raise RuntimeError('area id mismatch')

    def setup(self, env):
        for o in self.push_objects:
            new_object_id = env.add_object(o)
            o.object_id = new_object_id
        for a in self.push_areas:
            new_area_id = env.add_area(a)
            a.object_id = new_area_id

    def clean(self, env):
        for o in self.push_objects:
            env.remove_object(o.object_id)
        for a in self.push_areas:
            env.remove_area(a.object_id)