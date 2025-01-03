import numpy as np
import cv2
import copy
from bullet_env.env import BulletEnv
from transform.affine import Affine

from loguru import logger


class PushingEnv(BulletEnv):
    def __init__(
        self,
        debug,
        bullet_client,
        robot,
        task_factory,
        teletentric_camera,
        workspace_bounds,
        movement_bounds,
        step_size,
        gripper_offset,
        fixed_z_height,
        absolut_movement,
        distance_reward_scale,
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
        self.gripper_offset = Affine(gripper_offset.translation, gripper_offset.rotation)  # Affine([0, 0, 0], [3.14159265359, 0, 1.57079632679])
        self.fixed_z_height = fixed_z_height
        self.movement_punishment = False
        self.absolut_movement = absolut_movement
        self.distance_reward_scale = distance_reward_scale
        self.dist_list = []
        self.old_dist = []
        self.old_eef_pos = None
        self.debug = debug

    def reset(self):
        """Reset environment and return initial state"""
        # Clean up previous task if exists
        if self.current_task:
            self.current_task.clean(self)
            self.dist_list = []
            self.old_dist = []

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

        # Create initial distance list
        for i in range(len(self.current_task.push_objects)):
            self.dist_list.append(0.0)
            self.old_dist.append(0.0)

        # Reset eef pose
        self.old_eef_pos = self.robot.get_eef_pose().translation[:2]

        # Get initial observation
        return self._get_observation()

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        self.current_step += 1

        # Reset movement punishment flag
        self.movement_punishment = False

        if self.absolut_movement:
            # Normalize action between [0,1]
            action = (action + 1) / 2

            # Convert normalized action [0,1] to movement bounds
            x_range = self.movement_bounds[0][1] - self.movement_bounds[0][0]
            y_range = self.movement_bounds[1][1] - self.movement_bounds[1][0]

            x_move = action[0] * x_range + self.movement_bounds[0][0]
            y_move = action[1] * y_range + self.movement_bounds[1][0]

            logger.debug(f"Action: {action}, x_move: {x_move}, y_move: {y_move}")

            # Move robot
            new_pose = Affine([x_move, y_move, 0])

        else:
            # Convert normalized action [-1,1] to workspace movement
            x_range = self.workspace_bounds[0][1] - self.workspace_bounds[0][0]
            y_range = self.workspace_bounds[1][1] - self.workspace_bounds[1][0]

            x_move = action[0] * x_range * self.step_size
            y_move = action[1] * y_range * self.step_size

            logger.debug(f"Action: {action}, x_move: {x_move}, y_move: {y_move}")

            # Move robot
            current_pose = self.robot.get_eef_pose()
            new_pose = current_pose * Affine([x_move, y_move, 0])

            # Check if new pose is within movement bounds
            if (
                self.movement_bounds[0][0] <= new_pose.translation[0] <= self.movement_bounds[0][1]
                and self.movement_bounds[1][0] <= new_pose.translation[1] <= self.movement_bounds[1][1]
            ):
                logger.debug("New pose is within movement bounds.")
            else:
                logger.debug("New pose is outside movement bounds.")
                new_pose = current_pose  # Keep the current pose if the new pose is outside bounds
                self.movement_punishment = True

        # Maintain fixed height and orientation
        new_pose = Affine(translation=[new_pose.translation[0], new_pose.translation[1], self.fixed_z_height]) * self.gripper_offset

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
        logger.debug(f"Calculating reward for step {self.current_step}")

        for i in range(len(self.current_task.push_objects)):
            # Current objects and areas
            obj, area = self.current_task.get_object_and_area_with_same_id(i)
            obj_pose = self.get_pose(obj.unique_id)
            area_pose = self.get_pose(area.unique_id)

            # Distance-based reward
            obj_pos = obj_pose.translation[:2]
            area_pos = area_pose.translation[:2]
            self.dist_list[i] = np.linalg.norm(obj_pos - area_pos)

            # Calculate reward based on distance between current and previous step
            if self.current_step != 1:  # Skip first step because there is no previous step
                current_reward = round((self.old_dist[i] - self.dist_list[i]), 3) * self.distance_reward_scale
                total_reward += current_reward
                logger.debug(f"Distance reward for object {i}: {current_reward}")

            # placeholder for IoU

        # Copy current distance list for next step
        self.old_dist = copy.deepcopy(self.dist_list)

        # Slight reward for being within workspace bounds
        eef_pos = self.robot.get_eef_pose().translation[:2]
        if (
            self.workspace_bounds[0][0] <= eef_pos[0] <= self.workspace_bounds[0][1]
            and self.workspace_bounds[1][0] <= eef_pos[1] <= self.workspace_bounds[1][1]
        ):
            total_reward += 5.0
            logger.debug("Positive reward given for being within workspace bounds.")

        # Punish for moving outside movement bounds
        if self.movement_punishment:
            total_reward -= 5.0
            logger.debug("Negative reward -5.0 given for moving outside movement bounds.")

        # Punishment for not moving
        if np.linalg.norm(eef_pos - self.old_eef_pos) < 0.01 and self.current_step != 1:
            total_reward -= 5.0
            logger.debug("Negative reward -5.0 given for not moving.")

        # Update old eef position
        self.old_eef_pos = copy.deepcopy(eef_pos)

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
        logger.debug("Environment closed.")
