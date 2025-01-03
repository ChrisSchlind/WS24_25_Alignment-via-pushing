from transform.affine import Affine
from manipulation.simulation_object import SceneObject
from bullet_env.util import stdout_redirected, get_link_index

from manipulation.task.simple_push_task import PushArea


class BulletEnv:
    def __init__(self, bullet_client, coordinate_axes_urdf_path):
        self.bullet_client = bullet_client
        self.coordinate_axes_urdf_path = coordinate_axes_urdf_path
        self.coordinate_ids = []

    def add_object(self, o, scale=1) -> int:
        o_pose = Affine.from_matrix(o.pose)
        with stdout_redirected():
            obj_id = self.bullet_client.loadURDF(
                o.urdf_path,
                o_pose.translation,
                o_pose.quat,
                useFixedBase=o.static,
                globalScaling=scale,
                flags=self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            
        # Apply the color to all visual shapes of the object
        for i in range(self.bullet_client.getNumJoints(obj_id)):
            self.bullet_client.changeVisualShape(obj_id, i, rgbaColor=o.color)
        
        # Also apply the color to the base link (link index -1)
        self.bullet_client.changeVisualShape(obj_id, -1, rgbaColor=o.color)
        
        # Set the unique id of the object for later reference e.g. for removing the object
        o.unique_id = obj_id

    def remove_object(self, unique_id):
        with stdout_redirected():
            self.bullet_client.removeBody(unique_id)
        self.bullet_client.stepSimulation()

    def add_area(self, a, scale=1) -> int:
        a_pose = Affine.from_matrix(a.pose)
        with stdout_redirected():
            area_id = self.bullet_client.loadURDF(
                a.urdf_path,
                a_pose.translation,
                a_pose.quat,
                useFixedBase=a.static,
                globalScaling=scale,
                flags=self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
        # Apply the color to all visual shapes of the area
        for i in range(self.bullet_client.getNumJoints(area_id)):
            self.bullet_client.changeVisualShape(area_id, i, rgbaColor=a.color)
        
        # Also apply the color to the base link (link index -1)
        self.bullet_client.changeVisualShape(area_id, -1, rgbaColor=a.color)
        
        # Set the unique id of the area for later reference e.g. for removing the area
        a.unique_id = area_id

    def remove_area(self, unique_id):
        with stdout_redirected():
            self.bullet_client.removeBody(unique_id)
        self.bullet_client.stepSimulation() 

    def spawn_coordinate_frame(self, pose, scale=1):
        coordinate_axes = SceneObject(
            urdf_path=self.coordinate_axes_urdf_path,
            pose=pose,
        )
        c_id = self.add_object(coordinate_axes, scale)
        self.coordinate_ids.append(c_id)

    def remove_coordinate_frames(self):
        for c_id in self.coordinate_ids:
            self.remove_object(c_id)
        self.coordinate_ids = []

    def get_pose(self, unique_id: int):
        pos, quat = self.bullet_client.getBasePositionAndOrientation(unique_id)
        return Affine(pos, quat)
    
    def get_link_index(self, body_id: int, link_name: str):
        return get_link_index(self.bullet_client, body_id, link_name)


    def get_objects_intersection_volume(self, unique_id1: int, unique_id2: int):
        # Get the AABB (Axis-Aligned Bounding Box) for both objects
        aabb1 = self.bullet_client.getAABB(unique_id1)
        aabb2 = self.bullet_client.getAABB(unique_id2)

        # Calculate the intersection volume of the two AABBs
        intersection_min = [max(aabb1[0][i], aabb2[0][i]) for i in range(3)]
        intersection_max = [min(aabb1[1][i], aabb2[1][i]) for i in range(3)]

        # Check if there is an intersection
        if all(intersection_min[i] < intersection_max[i] for i in range(3)):
            intersection_volume = 1
            for i in range(3):
                intersection_volume *= (intersection_max[i] - intersection_min[i])
            return intersection_volume
        else:
            return 0
