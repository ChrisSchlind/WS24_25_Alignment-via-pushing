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
        iou_reward_scale,  # Add this parameter
        no_movement_threshold,
        max_moves_without_positive_reward,
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
        self.iou_reward_scale = iou_reward_scale  # Initialize the attribute
        self.no_movement_threshold = no_movement_threshold
        self.max_moves_without_positive_reward = max_moves_without_positive_reward
        self.dist_list = []
        self.old_dist = []
        self.iou_list = []
        self.old_iou = []
        self.old_eef_pos = None
        self.debug = debug
        self.moves_without_positive_reward = 0
        self.absolute_distances = []
        self.distance_rewards = []
        
    def reset(self):
        """Reset environment and return initial state"""
        # Clean up previous task if exists
        if self.current_task:
            self.current_task.clean(self)
            self.dist_list = []
            self.old_dist = []
            self.iou_list = []
            self.old_iou = []

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

        # Create initial distance and IoU lists
        for _ in range(len(self.current_task.push_objects)):
            self.dist_list.append(0.0)
            self.old_dist.append(0.0)
            self.iou_list.append(0.0)
            self.old_iou.append(0.0)
            self.absolute_distances.append(0.0)
            self.distance_rewards.append(0.0)

        # Reset eef pose
        self.old_eef_pos = self.robot.get_eef_pose().translation[:2]

        # Get initial observation
        return self._get_observation()

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        self.current_step += 1
        failed = False

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

        # Check if objects are inside workspace bounds
        if not self._check_objects():
            failed = True
            logger.debug("Objects are outside workspace bounds.")

        return next_state, reward, done, info, failed

    def _get_observation(self):
        """Get RGB + depth observation from orthographic camera"""
        obs = self.teletentric_camera.get_observation()

        # Resize RGB image (500x500x3 -> 84x84x3)
        rgb = cv2.resize(obs["rgb"], (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize RGB values between [0,1]
        rgb = rgb / 255.0

        # Resize and normalize depth image (500x500 -> 84x84)
        depth = obs["depth"]
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]  # Take first channel if depth is 3D
        depth = cv2.resize(depth, (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize depth values between [0,1] and prevent division by zero
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Add channel dimension to depth
        depth = depth[..., np.newaxis]

        # Concatenate RGB and depth into state
        state = np.concatenate([rgb, depth], axis=-1)

        # Ensure correct shape and type
        state = state.astype(np.float32)
        return state

    def _calculate_reward(self):
        """Calculate reward based on distance and iou."""
        total_reward = 0
        positive_reward_flag = False
        logger.debug(f"Calculating reward for step {self.current_step}")

        '''
        REWARDS:
        1. Distance-based reward
        2. IoU-based reward
        '''

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
                absolute_distance = round((self.old_dist[i] - self.dist_list[i]), 3)
                
                if absolute_distance > 0: # ignore negative movement, only reward positive movement
                    current_reward = absolute_distance * self.distance_reward_scale
                    self.moves_without_positive_reward = 0 # reset counter
                    positive_reward_flag = True
                else:
                    current_reward = 0.0

                total_reward += round(current_reward, 2)
                logger.debug(f"Distance reward for object {i}: {round(current_reward, 2)}")

                # Save distances and reward for DQNSupervisor
                self.absolute_distances[i] = absolute_distance
                self.distance_rewards[i] = current_reward

            # Calculate IoU-based reward (if IoU gets higher, reward is positive, scaled by self.iou_reward_scale)
            if self.current_step != 1:
                IOU = self.get_objects_intersection_volume(obj.unique_id, area.unique_id)
                #relative_iou = IOU - self.old_iou[i]
                absolute_iou = IOU * 10e6 # scale IoU to be greater than 0.1, currently all IoU values are XXXXe-09
                total_reward += round(absolute_iou * self.iou_reward_scale, 2)
                if absolute_iou > 0:
                    self.moves_without_positive_reward = 0 # reset counter
                    positive_reward_flag = True
                logger.debug(f"IoU reward for object {i}: {round(absolute_iou * self.iou_reward_scale, 2)}")


        # Copy current distance and IoU lists for next step
        self.old_dist = copy.deepcopy(self.dist_list)
        self.old_iou = copy.deepcopy(self.iou_list)        

        '''
        PUNISHMENTS:
        1. Moving outside movement bounds
        2. Not moving at all
        3. Not moving object or increasing IoU
        '''

        # Punish for moving outside movement bounds
        if self.movement_punishment:
            total_reward -= 10.0
            logger.debug("Negative reward -10.0 given for moving outside movement bounds.")

        # Punishment for not moving
        eef_pos = self.robot.get_eef_pose().translation[:2]
        if np.linalg.norm(eef_pos - self.old_eef_pos) < self.no_movement_threshold and self.current_step != 1:
            total_reward -= 100.0
            logger.debug("Negative reward -100.0 given for not moving.")
        # No positive reward for moving because than the robot will just move around without any purpose

        # Update old eef position
        self.old_eef_pos = copy.deepcopy(eef_pos)

        # Punishment for not moving object or increasing IoU
        if self.moves_without_positive_reward >= self.max_moves_without_positive_reward:
            penalty = 100 + (self.moves_without_positive_reward - self.max_moves_without_positive_reward) # increase penalty for every step after max_moves_without_positive_reward
            total_reward -= penalty
            logger.debug(f"Negative reward -{penalty} given for not moving object or increasing IoU for the last {self.moves_without_positive_reward} steps.")

        # if the agent is not moving an object or increasing the IoU, count up
        if not positive_reward_flag:
            self.moves_without_positive_reward += 1        

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
    
    def _check_objects(self, extra_distance=0.1):
        """Check if all the objects are inside the workspace bounds"""
        counter = 0
        for i in range(len(self.current_task.push_objects)):
            obj = self.current_task.push_objects[i]
            obj_pos = self.get_pose(obj.unique_id).translation[:2]

            if {obj_pos[0] < (self.workspace_bounds[0][0] - extra_distance) or 
                obj_pos[0] > (self.workspace_bounds[0][1] + extra_distance) or 
                obj_pos[1] < (self.workspace_bounds[1][0] - extra_distance) or 
                obj_pos[1] > (self.workspace_bounds[1][1] + extra_distance)}:
                counter += 1

        if counter == len(self.current_task.push_objects):
            return True

        return False

    def render(self):
        """Return the current camera view"""
        return self._get_observation()

    def close(self):
        """Close the environment"""
        self.bullet_client.disconnect()
        logger.debug("Environment closed.")

    def calculate_iou(self, mask1, mask2):
        """Calculate Intersection over Union (IoU) between two binary masks."""
        logger.debug(f"Calculating IoU between two masks of shape {mask1.shape} and {mask2.shape}")
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union if union != 0 else 0
        return iou


def get_object_mask(bullet_client, unique_id):
    """Generate a binary mask for the object."""
    width, height, _, _, segmentation_mask = bullet_client.getCameraImage(84, 84)
    mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if segmentation_mask[i, j] == unique_id:
                mask[i, j] = 1
    return mask
