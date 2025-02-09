import os
from typing import List, Dict, Any, Union
import random
import numpy as np
from dataclasses import dataclass, field
import json
from manipulation.simulation_object import SceneObject, is_overlapping
from transform.affine import Affine
from loguru import logger

class PushObjectFactory:
    def __init__(self, objects_root: str, object_types: Union[List[str], None] = None):
        self.objects_root = objects_root
        if object_types is None or len(object_types) == 0:
            self.object_types = [f.name for f in os.scandir(objects_root) if f.is_dir()]
        else:
            self.object_types = object_types

    def create_push_object(self, object_type: str):
        urdf_path = f"{self.objects_root}/{object_type}/object.urdf"
        kwargs = {"urdf_path": urdf_path}

        # Read push configuration
        push_config_path = f"{self.objects_root}/{object_type}/push_config.json"
        with open(push_config_path) as f:
            push_args = json.load(f)

        # Process offset transformation
        if "offset" in push_args:
            offset_data = push_args["offset"]
            translation = offset_data.get("translation", [0, 0, 0])
            rotation = offset_data.get("rotation", [0, 0, 0])
            offset = Affine(translation, rotation).matrix
            push_args["offset"] = offset

        kwargs.update(push_args)
        
        # Generate random color if not specified
        if "color" not in kwargs:
            kwargs["color"] = [random.uniform(0.3, 1.0) for _ in range(3)] + [1.0] # change from random.random() to random.uniform(0.3, 1.0), prevents dark colors

            # Check that color is not white or light gray
            while np.linalg.norm(kwargs["color"]) > 2.5:
                kwargs["color"] = [random.random() for _ in range(3)] + [1.0]        

        # For testing purposes, set color to red
        #kwargs["color"] = [0.8, 0.2, 0.2, 1.0]

        return PushObject(**kwargs)

@dataclass
class PushObject(SceneObject):
    """
    Class for objects that can be pushed. A push configuration is required.

    """

    static: bool = False
    push_config: List[Dict[str, Any]] = field(default_factory=lambda: [])
    color: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 1.0])

    def get_mask(self, bullet_client):
        """Generate a binary mask for the object."""
        width, height, _, _, segmentation_mask = bullet_client.getCameraImage(84, 84)
        mask = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if segmentation_mask[i, j] == self.unique_id:
                    mask[i, j] = 1
        return mask

class PushAreaFactory:
    def __init__(self, areas_root: str, areas_types: Union[List[str], None] = None):
        self.areas_root = areas_root
        if areas_types is None:
            self.areas_types = [f.name for f in os.scandir(areas_root) if f.is_dir()]
        else:
            self.areas_types = areas_types

    def create_push_area(self, area_type: str, color: List[float]):
        urdf_path = f"{self.areas_root}/{area_type}/area.urdf"
        kwargs = {"urdf_path": urdf_path}

        # Read push configuration
        push_config_path = f"{self.areas_root}/{area_type}/push_config.json"
        with open(push_config_path) as f:
            push_args = json.load(f)

        # Process offset transformation
        if "offset" in push_args:
            offset_data = push_args["offset"]
            translation = offset_data.get("translation", [0, 0, 0])
            rotation = offset_data.get("rotation", [0, 0, 0])
            offset = Affine(translation, rotation).matrix
            push_args["offset"] = offset

        kwargs.update(push_args)
        kwargs["color"] = color

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
    def __init__(self, n_objects: int, min_n_objects: int, t_bounds, r_bounds=None, push_object_factory=None, push_area_factory=None):
        self.n_objects = n_objects
        self.min_n_objects = min_n_objects
        self.t_bounds = np.array(t_bounds)
        self.r_bounds = np.array([[0, 0], [0, 0], [0, 2 * np.pi]]) if r_bounds is None else np.array(r_bounds)
        self.push_object_factory = push_object_factory
        self.push_area_factory = push_area_factory
        self._reset_counters()

    def _reset_counters(self):
        self.unique_id_counter = 0
        self.object_id_counter = 0
        self.area_id_counter = 0

    def generate_push_object(self, object_type, added_objects):
        # Create object
        obj = self.push_object_factory.create_push_object(object_type)

        # Find non-overlapping pose
        pose, found_pose = self.get_non_overlapping_pose(obj.min_dist, added_objects)

        # Apply offset and set pose
        obj.pose = obj.offset @ pose.matrix
        obj.unique_id = self._get_unique_id()
        obj.object_id = self._get_object_id()

        return obj, found_pose

    def generate_push_area(self, object_type, added_areas, push_object):
        # Create area with matching color
        area = self.push_area_factory.create_push_area(object_type, push_object.color)

        # Find non-overlapping pose
        areas_and_objects = added_areas + [push_object]
        pose, found_pose = self.get_non_overlapping_pose(area.min_dist, areas_and_objects)

        # Apply offset and set pose
        area.pose = area.offset @ pose.matrix
        area.unique_id = self._get_unique_id()
        area.area_id = self._get_area_id()

        return area, found_pose

    def _get_unique_id(self):
        self.unique_id_counter += 1
        return self.unique_id_counter - 1

    def _get_object_id(self):
        self.object_id_counter += 1
        return self.object_id_counter - 1

    def _get_area_id(self):
        self.area_id_counter += 1
        return self.area_id_counter - 1

    def create_task(self):
        self._reset_counters()
        n_objects = np.random.randint(self.min_n_objects, self.n_objects + 1)
        object_types = random.choices(self.push_object_factory.object_types, k=n_objects)
        
        push_objects = []
        push_areas = []
        for object_type in object_types:
            found_pose_obj = False
            found_pose_area = False
            push_object, found_pose_obj = self.generate_push_object(object_type, push_objects)
            push_area, found_pose_area = self.generate_push_area(object_type, push_areas, push_object)
            if found_pose_obj and found_pose_area:
                push_objects.append(push_object)
                push_areas.append(push_area)

        # Check if lists are empty
        if not push_objects or not push_areas:
            raise RuntimeError("Could not place a single object or area.")
        
        logger.debug(f"Generated {len(push_objects)}/{n_objects} objects and {len(push_areas)}/{n_objects} areas.")

        return PushTask(push_objects, push_areas)

    def get_non_overlapping_pose(self, min_dist, objects):
        counter = 0
        overlapping = True
        found_pose = True
        new_t_bounds = np.array(self.t_bounds)
        new_t_bounds[:2, 0] = new_t_bounds[:2, 0] + min_dist
        new_t_bounds[:2, 1] = new_t_bounds[:2, 1] - min_dist
        while overlapping:
            random_pose = Affine.random(t_bounds=new_t_bounds, r_bounds=self.r_bounds)
            overlapping = is_overlapping(random_pose, min_dist, objects)

            if counter > 1000:
                logger.info("Could not find non-overlapping pose.")
                found_pose = False
                return random_pose, found_pose

            counter += 1

        return random_pose, found_pose


class PushTask:
    def __init__(self, push_objects: List[PushObject], push_areas: List[PushArea]):
        self.push_objects = push_objects
        self.push_areas = push_areas

    def get_info(self):
        info = {"_target_": "manipulation.task.simple_push_task.PushTask", "push_objects": self.push_objects, "push_areas": self.push_areas}
        return info

    def get_object_with_unique_id(self, unique_id: int):
        for o in self.push_objects:
            if o.unique_id == unique_id:
                return o
        raise RuntimeError("object unique id mismatch")

    def get_area_with_unique_id(self, unique_id: int):
        for a in self.push_areas:
            if a.unique_id == unique_id:
                return a
        raise RuntimeError("area unique id mismatch")

    def get_object_with_matching_id(self, object_id: int):
        for o in self.push_objects:
            if o.object_id == object_id:
                return o
        raise RuntimeError("object id mismatch")

    def get_area_with_matching_id(self, area_id: int):
        for a in self.push_areas:
            if a.area_id == area_id:
                return a
        raise RuntimeError("area id mismatch")

    def get_object_and_area_with_same_id(self, id: int):
        obj = self.get_object_with_matching_id(id)
        area = self.get_area_with_matching_id(id)
        return obj, area

    def setup(self, env):
        for o in self.push_objects:
            env.add_object(o)
        for a in self.push_areas:
            env.add_area(a)

    def clean(self, env):
        for o in self.push_objects:
            env.remove_object(o.unique_id)
        for a in self.push_areas:
            env.remove_area(a.unique_id)

    def reset_env(self, env):
        self.clean(env)
        self.setup(env)
