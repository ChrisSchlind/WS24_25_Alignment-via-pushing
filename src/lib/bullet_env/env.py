from transform.affine import Affine
from manipulation.simulation_object import SceneObject
from bullet_env.util import stdout_redirected, get_link_index


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
        o.object_id = obj_id
        return obj_id

    def remove_object(self, object_id):
        with stdout_redirected():
            self.bullet_client.removeBody(object_id)
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

    def get_object_pose(self, object_id: int):
        pos, quat = self.bullet_client.getBasePositionAndOrientation(object_id)
        return Affine(pos, quat)
    
    def get_link_index(self, body_id: int, link_name: str):
        return get_link_index(self.bullet_client, body_id, link_name)
