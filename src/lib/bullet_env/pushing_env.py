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
        distance_TCP_obj_reward_scale,
        distance_obj_area_reward_scale,
        iou_reward_scale,  # Add this parameter
        no_movement_threshold,
        max_moves_without_positive_reward,
        success_threshold_trans,
        success_threshold_rot,
        activate_distance_obj_area_reward,
        activate_distance_TCP_obj_reward,
        activate_iou_reward,
        activate_moves_without_positive_reward,
        activate_no_movement_punishment,
        activate_objects_outside_workspace_punishment,
        angle_obj_area_tcp_threshold,
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
        self.success_threshold_trans = success_threshold_trans
        self.success_threshold_rot = success_threshold_rot
        self.max_steps = max_steps
        self.current_step = 0
        self.current_task = None
        self.gripper_offset = Affine(gripper_offset.translation, gripper_offset.rotation)  # Affine([0, 0, 0], [3.14159265359, 0, 1.57079632679])
        self.fixed_z_height = fixed_z_height
        self.movement_punishment = False
        self.absolut_movement = absolut_movement
        self.distance_TCP_obj_reward_scale = distance_TCP_obj_reward_scale
        self.distance_obj_area_reward_scale = distance_obj_area_reward_scale
        self.iou_reward_scale = iou_reward_scale
        self.no_movement_threshold = no_movement_threshold
        self.max_moves_without_positive_reward = max_moves_without_positive_reward
        self.activate_distance_obj_area_reward = activate_distance_obj_area_reward
        self.activate_distance_TCP_obj_reward = activate_distance_TCP_obj_reward
        self.activate_iou_reward = activate_iou_reward
        self.activate_moves_without_positive_reward = activate_moves_without_positive_reward
        self.activate_no_movement_punishment = activate_no_movement_punishment
        self.activate_objects_outside_workspace_punishment = activate_objects_outside_workspace_punishment
        self.angle_obj_area_tcp_threshold = angle_obj_area_tcp_threshold
        self.dist_list = []
        self.old_dist = []
        self.iou_list = []
        self.old_iou = []
        self.old_eef_pos = None
        self.debug = debug
        self.moves_without_positive_reward = 0
        self.absolute_TCP_obj_distances = []
        self.distance_TCP_obj_rewards = []
        self.absolute_obj_area_distances = []
        self.distance_obj_area_rewards = []
        self.absolute_distance_TCP_obj_last = []

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

        self.robot.home()  # Move robot to home position

        # start pose = a random position in the workspace
        start_pose = Affine(
            translation=[
                np.random.uniform(self.workspace_bounds[0][0], self.workspace_bounds[0][1]),
                np.random.uniform(self.workspace_bounds[1][0], self.workspace_bounds[1][1]),
                self.fixed_z_height,
            ]
        )

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
            self.absolute_TCP_obj_distances.append(0.0)
            self.distance_TCP_obj_rewards.append(0.0)
            self.absolute_obj_area_distances.append(0.0)
            self.distance_obj_area_rewards.append(0.0)
            self.absolute_distance_TCP_obj_last.append(0.0)  # Initialize the attribute

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

        # Check if objects are outside workspace bounds
        if not self._check_objects():
            failed = True
            logger.info("Objects are outside workspace bounds.")

        # Check if robot got himself into a weird position (e.g. axis 3 is on the ground)
        if self.robot.get_eef_pose().translation[2] > (30 * self.fixed_z_height):
            failed = True
            logger.info("Robot crashed himself trying to reach a non-optimal position. Resetting environment and stop this episode.")

        return next_state, reward, done, info, failed

    def _get_observation(self):
        """Get RGB + depth observation from orthographic camera"""
        obs = self.teletentric_camera.get_observation()

        # Resize RGB image (500x500x3 -> 88x88x3)
        rgb = cv2.resize(obs["rgb"], (88, 88), interpolation=cv2.INTER_AREA)

        # Normalize RGB values between [0,1]
        rgb = rgb / 255.0

        # Resize and normalize depth image (500x500 -> 88x88)
        depth = obs["depth"]
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]  # Take first channel if depth is 3D
        depth = cv2.resize(depth, (88, 88), interpolation=cv2.INTER_AREA)

        # Note: Depth values are already normalized between [0,1] in the camera class

        # Add channel dimension to depth
        depth = depth[..., np.newaxis]

        # Concatenate RGB and depth into state
        state = np.concatenate([rgb, depth, depth, depth], axis=-1)

        # Ensure correct shape and type
        state = state.astype(np.float32)
        return state

    def _calculate_reward(self):
        """Calculate reward based on distance and iou."""
        total_reward = 0
        positive_reward_flag = False
        logger.debug(f"Calculating reward for step {self.current_step}")

        """
        REWARDS:
        1. Distance-based reward of object to area
        2. IoU-based reward
        3. if(relativBewegungen): Distance-based reward of TCP to object


        Idea:
        Reduce the amount of rewards to just one per step, so that the agent is not overwhelmed by rewards. And can focus on the current task.
        """

        for i in range(len(self.current_task.push_objects)):
            # Initialize reward for each object
            current_reward = 0.0

            # Current objects and areas
            obj, area = self.current_task.get_object_and_area_with_same_id(i)
            obj_pose = self.get_pose(obj.unique_id)
            area_pose = self.get_pose(area.unique_id)

            # ******************************************************************
            # Distance-based reward of object to area
            if self.activate_distance_obj_area_reward:
                obj_pos = obj_pose.translation[:2]
                area_pos = area_pose.translation[:2]
                self.dist_list[i] = np.linalg.norm(obj_pos - area_pos)

                # Calculate reward based on distance between current and previous step
                absolute_distance_obj_area = round((self.old_dist[i] - self.dist_list[i]), 3)  # Ensure assignment

                if self.current_step != 1:  # Skip first step because there is no previous step
                    if absolute_distance_obj_area > 0: 
                        current_reward = absolute_distance_obj_area * self.distance_obj_area_reward_scale
                        self.moves_without_positive_reward = 0  # reset counter
                        positive_reward_flag = True
                    else:
                        current_reward = 0.0

                    # Apply curve to make very high rewards gradually smaller. This is so that accidental high rewards do not dominate the training but even small movements are rewarded
                    if abs(current_reward) > 40:
                        current_reward = 40 + (current_reward - 40) * 0.5 if current_reward > 0 else -40 + (current_reward + 40) * 0.5
                    elif abs(current_reward) > 15:
                        current_reward = 15 + (current_reward - 15) * 0.75 if current_reward > 0 else -15 + (current_reward + 15) * 0.75

                    total_reward += round(current_reward, 2)

                    if current_reward != 0:
                        logger.info(f"Distance reward for object-area {i}: {round(current_reward, 2)}")
                    else:
                        logger.debug(f"Distance reward for object-area {i}: {round(current_reward, 2)}")

                    # Save distances and reward for DQNSupervisor
                    self.absolute_obj_area_distances[i] = absolute_distance_obj_area
                    self.distance_obj_area_rewards[i] = current_reward

            # ******************************************************************
            # Calculate IoU-based reward (if IoU gets higher, reward is positive, scaled by self.iou_reward_scale)
            if self.activate_iou_reward:  # Reward only when it's activated
                IOU = self.get_objects_intersection_volume(obj.unique_id, area.unique_id)
                iou_delta = IOU - self.old_iou[i]
                if self.current_step != 1:  # Skip the first step since old_iou is initially 0
                    if iou_delta > 0:  # IOU got better -> positive reward
                        current_reward = iou_delta * 1e6 * self.iou_reward_scale
                        total_reward += round(current_reward, 2)
                        self.moves_without_positive_reward = 0
                        positive_reward_flag = True
                        logger.info(f"IOU improvement reward for object {i}: {round(current_reward, 2)}")
                    elif iou_delta < 0:  # IOU got worse -> currently no punishment
                        # current_reward = iou_delta * 1e6 * self.iou_reward_scale * 0.1      # in this case iou_delta is negative, so the reward is negative and therefore the agent is punsihed
                        current_reward = 0.0  # currently no punishment for decreasing IoU
                        total_reward += round(current_reward, 2)
                        logger.debug(f"IOU decreased for object {i}: {round(current_reward, 2)}")

                # Update old_iou for next step
                self.old_iou[i] = IOU

            # ******************************************************************
            # Distance-based reward of TCP to object
            if self.activate_distance_TCP_obj_reward:
                tcp_pos = self.robot.get_eef_pose().translation[:2]
                obj_pos = obj_pose.translation[:2]
                area_pos = area_pose.translation[:2]

                # Calculate reward if the TCP got closer to any of the objects
                absolute_distance_TCP_obj_new = round(np.linalg.norm(tcp_pos - obj_pos), 3)

                if self.current_step == 1:  # Initialize during the first step
                    self.absolute_distance_TCP_obj_last[i] = absolute_distance_TCP_obj_new

                # Delta = new distance - last distance
                absolute_distance_delta = absolute_distance_TCP_obj_new - self.absolute_distance_TCP_obj_last[i]
                self.absolute_distance_TCP_obj_last[i] = absolute_distance_TCP_obj_new

                if absolute_distance_TCP_obj_new > obj.min_dist: # Check if the distance is greater than the min_dist

                    if absolute_distance_delta < 0: # TCP is moving towards the object
                        current_reward = -1.0 * absolute_distance_delta * self.distance_TCP_obj_reward_scale
                    else:
                        current_reward = (
                            -2.0 * absolute_distance_delta * self.distance_TCP_obj_reward_scale
                        )  # NO"*0.25" HERE!! = otherwise farming of rewards is easily possible by just moving back and forth --> this way the agent is forced to move to the object
                        
                else:
                    if self._check_angle_obj_tcp_area(area_pos, obj_pos, tcp_pos): # check if tcp is behind the object relative to the area

                        logger.info(f"TCP is behind the object {i} relative to the area {i}.")

                        if absolute_distance_delta < 0: # high positive reward if the TCP is behind the object and TCP is moving to object
                            current_reward = -20.0 * absolute_distance_delta * self.distance_TCP_obj_reward_scale
                        else:
                            current_reward = 0.0

                    else:
                        current_reward = -10.0 

                total_reward += round(current_reward, 2)

                if current_reward != 0:
                    logger.info(f"Distance reward for TCP-object {i}: {round(current_reward, 2)}")
                else:
                    logger.debug(f"Distance reward for TCP-object {i}: {round(current_reward, 2)}")

                # Save distances and reward for DQNSupervisor
                self.absolute_TCP_obj_distances[i] = absolute_distance_delta  # Correct assignment
                self.distance_TCP_obj_rewards[i] = current_reward

        # Copy current distance and IoU lists for next step
        self.old_dist = copy.deepcopy(self.dist_list)
        self.old_iou = copy.deepcopy(self.iou_list)

        """
        PUNISHMENTS:
        1. Not moving at all
        2. Not moving object or increasing IoU
        3. Objects are outside workspace bounds
        """

        # Punishment for not moving
        if self.activate_no_movement_punishment:
            eef_pos = self.robot.get_eef_pose().translation[:2]
            if np.linalg.norm(eef_pos - self.old_eef_pos) < self.no_movement_threshold and self.current_step != 1:
                total_reward -= 100.0
                logger.info("Negative reward -100.0 given for not moving.")
            # No positive reward for moving because than the robot will just move around without any purpose

            # Update old eef position
            self.old_eef_pos = copy.deepcopy(eef_pos)

        # Punishment for not moving object or increasing IoU
        if self.activate_moves_without_positive_reward:
            if self.moves_without_positive_reward >= self.max_moves_without_positive_reward:
                penalty = (
                    20 + (self.moves_without_positive_reward - self.max_moves_without_positive_reward) * 3
                )  # increase penalty for every step after max_moves_without_positive_reward
                total_reward -= penalty
                logger.info(
                    f"Negative reward -{penalty} given for not moving object or increasing IoU for the last {self.moves_without_positive_reward} steps."
                )

            # if the agent is not moving an object or increasing the IoU, count up
            if not positive_reward_flag:
                self.moves_without_positive_reward += 1

        # Punishment for objects outside workspace bounds
        if self.activate_objects_outside_workspace_punishment:
            if not self._check_objects():
                total_reward -= 100.0
                logger.info("Negative reward -100.0 given for objects outside workspace bounds.")

        return total_reward
    
    def _check_angle_obj_tcp_area(self, pos_area, pos_obj, pos_tcp):
        rel_pos = np.array(pos_area) - np.array(pos_obj)
        vec_obj_to_tcp = np.array(pos_tcp) - np.array(pos_obj)

        theta = np.arctan2(rel_pos[1], rel_pos[0]) - np.arctan2(vec_obj_to_tcp[1], vec_obj_to_tcp[0])
        
        theta = np.mod(theta, 2 * np.pi)

        theta = theta * 180.0 / np.pi # Convert to degrees
        logger.debug(f"Angle between object and TCP with area {pos_area}: {theta:.2f} degrees")

        # Check if angle is within threshold
        if ((180.0 - self.angle_obj_area_tcp_threshold) <= theta <= (180.0 + self.angle_obj_area_tcp_threshold)):
            return True
        else:
            return False

    def _check_done(self):
        """Check if episode should end"""
        # Episode ends if max steps reached
        if self.current_step >= self.max_steps:
            return True

        # Check if all objects are in their areas with the correct orientation
        for i in range(len(self.current_task.push_objects)):
            obj, area = self.current_task.get_object_and_area_with_same_id(i)
            obj_pos = self.get_pose(obj.unique_id).translation[:2]
            area_pos = self.get_pose(area.unique_id).translation[:2]

            if np.linalg.norm(obj_pos - area_pos) > self.success_threshold_trans:
                # logger.debug(f"Object {i} is not in its area with {np.linalg.norm(obj_pos - area_pos):.2f} distance. Needed distance: {self.success_threshold_trans}.")
                return False

            if not self._check_object_to_area_rotation(obj, area):
                # logger.debug(f"Object {i} is not aligned with its area.")
                return False

        logger.info("All objects are in their areas.")

        # All objects are aligned
        return True

    def _check_objects(self, extra_distance=0.0):
        """Check if all the objects are inside the workspace bounds"""
        counter = 0
        for i in range(len(self.current_task.push_objects)):
            obj = self.current_task.push_objects[i]
            obj_pos = self.get_pose(obj.unique_id).translation[:2]

            if (
                (obj_pos[0] < (self.workspace_bounds[0][0] - extra_distance))
                or (obj_pos[0] > (self.workspace_bounds[0][1] + extra_distance))
                or (obj_pos[1] < (self.workspace_bounds[1][0] - extra_distance))
                or (obj_pos[1] > (self.workspace_bounds[1][1] + extra_distance))
            ):
                counter += 1

        if counter == len(self.current_task.push_objects):
            return False

        return True

    def _calculate_angle_between_rotations(self, rot1, rot2):
        """
        Calculate the angle between two rotation matrices.

        :param rot1: Rotation matrix of the object (3x3)
        :param rot2: Rotation matrix of the area (3x3)
        :return: Angle in degrees
        """
        # Relative rotation matrix
        relative_rot = np.dot(np.linalg.inv(rot1), rot2)

        # Extract the angle from the trace of the relative rotation matrix
        angle_rad = np.arccos((np.trace(relative_rot) - 1) / 2)

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def _check_object_to_area_rotation(self, obj, area):
        """
        Check if the object is aligned with the area.

        :param obj: Object to check
        :param area: Area to check
        :return: True if aligned, False otherwise
        """
        obj_rot = self.get_pose(obj.unique_id).rotation
        area_rot = self.get_pose(area.unique_id).rotation
        angle = self._calculate_angle_between_rotations(obj_rot, area_rot)

        # Use symmetry axis of object
        if obj.sym_axis > 0:
            angle = angle % int(180 / obj.sym_axis)
        else:  # e.g. round objects have inf. symmetry axis and are always aligned with the area, check value is -1
            return True

        if obj.sym_axis == 1:
            # logger.debug(f"Angle between object and area with id {obj.unique_id}: {angle:.2f} degrees with min/max: {0.00:.2f} degrees")
            return angle <= self.success_threshold_rot
        else:
            # logger.debug(f"Angle between object and area with id {obj.unique_id}: {angle:.2f} degrees with min: {0.00:.2f} and max: {180.0 / obj.sym_axis:.2f} degrees")
            return (
                angle <= self.success_threshold_rot or ((180.0 / obj.sym_axis) - angle) <= self.success_threshold_rot
            )  # Check if angle is within threshold for both directions

    def render(self):
        """Return the current camera view"""
        return self._get_observation()

    def close(self):
        """Close the environment"""
        self.bullet_client.disconnect()
        logger.debug("Environment closed.")

    def get_tcp_pose(self):
        """Get the current pose of the TCP (Tool Center Point)"""
        return self.robot.get_eef_pose()
