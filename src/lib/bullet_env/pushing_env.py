import numpy as np
import cv2
from bullet_env.env import BulletEnv
from transform.affine import Affine

from loguru import logger


class PushingEnv(BulletEnv):
    def __init__(
        self,
        bullet_client,
        robot,
        task_factory,
        teletentric_camera,
        workspace_bounds,
        movement_bounds,
        step_size,
        gripper_offset,
        fixed_z_height,
        success_threshold=0.05,
        max_steps=200,        
        coordinate_axes_urdf_path=None,
    ):
        super().__init__(bullet_client, coordinate_axes_urdf_path)
        self.robot = robot
        self.task_factory = task_factory
        self.teletentric_camera = teletentric_camera
        self.workspace_bounds = workspace_bounds
        self.movement_bounds = movement_bounds
        self.step_size = step_size
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        self.current_step = 0
        self.current_task = None
        self.gripper_offset = Affine(gripper_offset.translation, gripper_offset.rotation) #Affine([0, 0, 0], [3.14159265359, 0, 1.57079632679])
        self.fixed_z_height = fixed_z_height
        self.movement_punishment = False

    def reset(self):
        """Reset environment and return initial state"""
        # Clean up previous task if exists
        if self.current_task:
            self.current_task.clean(self)

        # Create new task and set up environment
        self.current_task = self.task_factory.create_task()
        self.current_task.setup(self)

        # Reset robot position
        self.robot.home()
        start_pose = Affine(translation=[0.35, -0.25, self.fixed_z_height])
        start_pose = start_pose * self.gripper_offset
        self.robot.ptp(start_pose)

        # Reset step counter
        self.current_step = 0

        # Get initial observation
        return self._get_observation()

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        self.current_step += 1
        self.movement_punishment = False

        # Convert normalized action [-1,1] to workspace movement
        x_range = self.workspace_bounds[0][1] - self.workspace_bounds[0][0]
        y_range = self.workspace_bounds[1][1] - self.workspace_bounds[1][0]

        x_move = action[0] * x_range * self.step_size
        y_move = action[1] * y_range * self.step_size

        logger.info(f"Action: {action}, x_move: {x_move}, y_move: {y_move}")

        # Move robot
        current_pose = self.robot.get_eef_pose()
        new_pose = current_pose * Affine([x_move, y_move, 0])

        # Check if new pose is within movement bounds
        if (
            self.movement_bounds[0][0] <= new_pose.translation[0] <= self.movement_bounds[0][1]
            and self.movement_bounds[1][0] <= new_pose.translation[1] <= self.movement_bounds[1][1]
        ):
            logger.info("New pose is within movement bounds.")
        else:
            logger.info("New pose is outside movement bounds.")
            new_pose = current_pose  # Keep the current pose if the new pose is outside bounds
            self.movement_punishment = True

        # Maintain fixed height and orientation
        new_pose = Affine(translation=[new_pose.translation[0], new_pose.translation[1], self.fixed_z_height]) * self.gripper_offset

        logger.info(f"Moving to {new_pose.translation} with orientation {new_pose.quat} for action {action}")
        logger.info(f"Current pose: {current_pose.translation} with orientation {current_pose.quat}")

        # Execute movement
        self.robot.ptp(new_pose)
        self.bullet_client.stepSimulation()

        # Get new observation
        next_state = self._get_observation()

        # Calculate reward and check if done
        reward = self._calculate_reward()
        done = self._check_done()

        # Include additional info
        info = {"current_step": self.current_step, "max_steps": self.max_steps}

        return next_state, reward, done, info

    def _get_observation(self):
        """Get RGB + depth observation from orthographic camera"""
        obs = self.teletentric_camera.get_observation()

        # Resize RGB image (500x500x3 -> 84x84x3)
        rgb = cv2.resize(obs["rgb"], (84, 84), interpolation=cv2.INTER_AREA)

        # Resize and normalize depth image (500x500 -> 84x84)
        depth = obs["depth"]
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]  # Take first channel if depth is 3D
        depth = cv2.resize(depth, (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize depth values
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Add channel dimension to depth
        depth = depth[..., np.newaxis]

        # Concatenate RGB and depth into state
        state = np.concatenate([rgb, depth], axis=-1)

        # Ensure correct shape and type
        state = state.astype(np.float32)
        return state

    def _calculate_reward(self):
        """Calculate reward based on distance and orientation."""
        total_reward = 0

        for i in range(len(self.current_task.push_objects)):
            obj, area = self.current_task.get_object_and_area_with_same_id(i)
            obj_pose = self.get_pose(obj.unique_id)
            area_pose = self.get_pose(area.unique_id)

            # Distance-based reward
            obj_pos = obj_pose.translation[:2]
            area_pos = area_pose.translation[:2]
            dist = np.linalg.norm(obj_pos - area_pos)
            total_reward -= dist  # negative distance to reduce it
            if dist < self.success_threshold:
                total_reward += 10.0

            # Orientation-based term as a placeholder for IoU:
            obj_yaw = np.arctan2(
                2.0 * (obj_pose.quat[3] * obj_pose.quat[2] + obj_pose.quat[0] * obj_pose.quat[1]),
                1.0 - 2.0 * (obj_pose.quat[1] ** 2 + obj_pose.quat[2] ** 2),
            )
            area_yaw = np.arctan2(
                2.0 * (area_pose.quat[3] * area_pose.quat[2] + area_pose.quat[0] * area_pose.quat[1]),
                1.0 - 2.0 * (area_pose.quat[1] ** 2 + area_pose.quat[2] ** 2),
            )
            yaw_diff = abs(obj_yaw - area_yaw)
            orientation_reward = 1.0 - min(yaw_diff / np.pi, 1.0)
            total_reward += orientation_reward  # simple orientation reward

        # Slight reward for being within workspace bounds
        eef_pos = self.robot.get_eef_pose().translation[:2]
        if (
            self.workspace_bounds[0][0] <= eef_pos[0] <= self.workspace_bounds[0][1]
            and self.workspace_bounds[1][0] <= eef_pos[1] <= self.workspace_bounds[1][1]
        ):
            total_reward += 0.5
            print("Positive reward given for being within workspace bounds.")
        
        # Check for collision between end effector and push objects
        for obj in self.current_task.push_objects:
            contact_points = self.bullet_client.getContactPoints(self.robot.eef_id, obj.unique_id)
            if contact_points:
                total_reward += 0.1  # Positive reward for collision
                logger.info(f"Collision detected between end effector and object {obj.unique_id}")

        # Small reward for making contact with objects
        for obj in self.current_task.push_objects:
            obj_pos = self.get_pose(obj.unique_id).translation[:2]
            if np.linalg.norm(eef_pos - obj_pos) < 0.07:  # Contact threshold, needs to be greater than maximum min_dist over all objects
                                                            # plus radius of the psuh cylinder
                total_reward += 0.05
                print("Positive reward given for making contact with an object.") 

        # Punish for moving outside movement bounds
        if self.movement_punishment:
            total_reward -= 5.0
            print("Negative reward given for moving outside movement bounds.")          

        return total_reward

    def _check_done(self):
        """Check if episode should end"""
        # Episode ends if max steps reached
        if self.current_step >= self.max_steps:
            return True

        # Check if all objects are in their areas
        for i in range(len(self.current_task.push_objects)):
            obj, area = self.current_task.get_object_and_area_with_same_id(i)
            obj_pos = self.get_pose(obj.unique_id).translation[:2]
            area_pos = self.get_pose(area.unique_id).translation[:2]

            if np.linalg.norm(obj_pos - area_pos) > self.success_threshold:
                return False

        # All objects are aligned
        return True

    def render(self):
        """Return the current camera view"""
        return self._get_observation()
    
    def close(self):
        """Close the environment"""
        self.bullet_client.disconnect()
        logger.info("Environment closed.")
