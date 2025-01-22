import sys
import copy  # Add this import
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import tensorflow as tf
import random
from collections import deque
import cv2
import os
from matplotlib import pyplot as plt
from bullet_env.util import setup_bullet_client, stdout_redirected
from transform.affine import Affine
from bullet_env.ur10_cell import UR10Cell  # Import UR10Cell
from bullet_env.pushing_env import PushingEnv  # Add this import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transporter_network.resnet import ResNet
import scipy.ndimage


class ConvDQN(tf.keras.Model):
    def __init__(self, initializer=tf.keras.initializers.GlorotUniform()):
        super().__init__()
        # Initializer for the weights
        self.initializer = initializer

        # ResNet Block 1: 
        self.resnet_block_1 = ResNet(kernel_size=(3, 3), output_depth=64, include_batchnorm=True)

        # ResNet Block 2: 
        self.resnet_block_2 = ResNet(kernel_size=(3, 3), output_depth=256, include_batchnorm=True)        

        # Conv2D layer for the heatmap output (H, W, 1)
        self.heatmap = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', kernel_initializer=self.initializer)

    def call(self, inputs, **kwargs):
        # Input shape: (batch_size, 88, 88, 6)

        # First ResNet-Block
        x = self.resnet_block_1(inputs)

        # Second ResNet-Block
        x = self.resnet_block_2(x)
     
        # Final heatmap (H, W, 1)
        x = self.heatmap(x)

        # Delete the last 2 rows and columns to get the correct heatmap size and remove ResNet errors at the edges
        x = tf.keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))(x)

        # Return the heatmap (no activation because this is a continuous value map)
        return x  # Heatmap of dimension (batch_size, 84, 84, 1)

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.losses = deque(maxlen=capacity)  # Store losses instead of priorities
        self.alpha = alpha

    # Add experience to the buffer with an initial high loss
    def put(self, state, action, reward, next_state, done, initial_loss=100.0):
        self.buffer.append([state, action, reward, next_state, done])
        self.losses.append(initial_loss)

    # Sample a batch of experiences from the buffer based on loss
    def sample(self, batch_size, beta=0.4, reward_range=(-1.0, 1.0)):
        losses = np.array(self.losses)
        losses[-1] = np.max(losses[:-1]) # set the newest losses to the same magnitude as the previous ones
        prob_dist = losses / np.sum(losses)
        
        reward_min = 0
        reward_max = 0

        # Sample half of the batch randomly and the other half based on the priority distribution
        half_batch_size = batch_size // 2
        indices_random = np.random.choice(len(self.buffer), half_batch_size, replace=False)
        indices_prob = np.random.choice(len(self.buffer), batch_size - half_batch_size, p=prob_dist)
        indices = np.concatenate((indices_random, indices_prob))

        weights = (len(self.buffer) * prob_dist[indices]) ** (-beta)
        weights /= weights.max()

        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

        reward_min = np.min(rewards) + 1e-8 # Add small value to avoid division by zero
        reward_max = np.max(rewards)

        # Normalize the rewards to the desired range
        # Formula: norm_reward = (reward - min) / (max - min) * (range_max - range_min) + range_min
        reward_min_new, reward_max_new = reward_range
        rewards_normalized = (rewards - reward_min) / (reward_max - reward_min) * (reward_max_new - reward_min_new) + reward_min_new

        return states, actions, rewards_normalized, next_states, dones, indices, weights

    # Update losses of sampled experiences
    def update_losses(self, indices, new_losses):
        for idx, new_loss in zip(indices, new_losses):
            self.losses[idx] = new_loss

    # Get the size of the buffer
    def size(self):
        return len(self.buffer)

class DQNSupervisor:
    def __init__(
        self,
        action_dim,
        env,
        workspace_bounds,
        min_obj_area_threshold=0.04,
        max_obj_area_threshold=0.6,
        extra_distance=0.05,
        sv_90deg_movement_threshold=0.1,
    ):
        self.action_dim = action_dim
        self.env = env
        self.workspace_bounds = workspace_bounds
        self.min_obj_area_threshold = min_obj_area_threshold
        self.max_obj_area_threshold = max_obj_area_threshold
        self.extra_distance = extra_distance
        self.sv_90deg_movement_threshold = sv_90deg_movement_threshold
        self.last_id = 0

    def ask_supervisor(self):
        # Initialize action
        rel_action = np.zeros(self.action_dim)
        action = np.zeros(self.action_dim)

        # Initialize id
        id = 0

        # Determine supervisor action as relative movement towards the object
        # After that convert relative movement to absolute movement

        # ----------------- Relative Movement -----------------

        # Calculate movement towards the object
        tcp_pose = self.env.get_tcp_pose()
        tcp_pos = tcp_pose.translation[:2]

        # Check for the nearest object to the TCP
        min_distance = float("inf")
        for i in range(len(self.env.current_task.push_objects)):
            obj, area = self.env.current_task.get_object_and_area_with_same_id(i)
            obj_pose = self.env.get_pose(obj.unique_id)
            area_pose = self.env.get_pose(area.unique_id)
            obj_pos = obj_pose.translation[:2]
            area_pos = area_pose.translation[:2]

            # skip objects that are already in the area and aligned with the area, important if more than one object is in the task
            if (
                np.linalg.norm(obj_pos - area_pos) > self.env.success_threshold_trans 
                and self.env._check_object_to_area_rotation(obj, area)
            ):
                continue

            dist = np.linalg.norm(obj_pos - tcp_pos)
            if dist < min_distance:
                min_distance = dist
                id = i

        obj, area = self.env.current_task.get_object_and_area_with_same_id(id)            
        obj_pose = self.env.get_pose(obj.unique_id)
        area_pose = self.env.get_pose(area.unique_id)
        obj_pos = obj_pose.translation[:2]
        area_pos = area_pose.translation[:2]

        # Movement/Distance&Direction between TCP and object
        movement = obj_pos - tcp_pos

        # Calculate orthogonal distance between the line (TCP, Area) and the object
        orthogonal_distance = self.orthogonal_distance(obj_pos, area_pos, tcp_pos)

        # Calculate distance between the object and the area
        obj_area_distance = np.linalg.norm(obj_pos - area_pos)

        # Calculate the random threshold for 90 degree movement
        random_90degDistance_threshold = self.sv_90deg_movement_threshold * random.uniform(0.8, 1.2)
        logger.debug(f"Movement distance: {np.linalg.norm(movement)}, tolerance: {random_90degDistance_threshold}")

        if orthogonal_distance < (obj.min_dist / 2):
            logger.debug(f"Supervisor said: Object is in line between TCP and Area and therefore move to object.")

            # Randomize the movement to the object based on the distance between the object and the area
            rel_action[1] = movement[1] * random.uniform((0.2 + abs(obj_area_distance)), (1.0 + abs(obj_area_distance)))
            rel_action[0] = movement[0] * random.uniform((0.2 + abs(obj_area_distance)), (1.0 + abs(obj_area_distance)))

        elif np.linalg.norm(movement) < random_90degDistance_threshold:
            # If the object is not in line between TCP and Area, move to the side of the object
            # If movement is too small, take an action weighted to the side of the object  

            logger.debug(f"Supervisor said: TCP close to the object, doing a mvt. 90° CCW")
            movement = np.array([movement[1], -movement[0]])

            # Randomize the movement to the side of the object
            rel_action[1] = movement[1] * random.uniform(1, 1.2)
            rel_action[0] = movement[0] * random.uniform(1, 1.2)
        
        else:
            # Randomize the movement to the object and make it smaller so the supervisor does not push the object of the table
            logger.debug(f"Supervisor said: Move to the object.")

            # Randomize the movement to the object
            rel_action[1] = movement[1] * random.uniform(0.7, 0.9)
            rel_action[0] = movement[0] * random.uniform(0.7, 0.9)

        # ----------------- Absolute Movement -----------------

        # Convert relative movement to absolute movement
        abs_action = tcp_pos + rel_action

        # Define the range of the movement space
        x_range = self.env.movement_bounds[0][1] - self.env.movement_bounds[0][0]
        y_range = self.env.movement_bounds[1][1] - self.env.movement_bounds[1][0]

        # Normalize action in movement boundaries
        action[0] = (abs_action[0] - self.env.movement_bounds[0][0]) / x_range
        action[1] = (abs_action[1] - self.env.movement_bounds[1][0]) / y_range

        action = 2 * action - 1 # to convert from [0,1] to [-1,1]

        # Clip action between [-1 1], needed for objects that are pushed off the table and now lie outside of workspace in the void
        action = np.clip(action, -1, 1)

        logger.debug(f"Supervisor said: Action {action} for object pose: {obj_pos}")

        return action
    
    def orthogonal_distance(self, obj_pos, area_pos, tcp_pos):
        """
        Calculates the orthogonal distance between a line and a point.

        :param obj_pos: List [x, y] of the object's position
        :param area_pos: List [x, y] of the area's position
        :param tcp_pos: List [x, y] of the TCP's position
        :return: Orthogonal distance
        """
        # Convert to numpy arrays for easier calculations
        obj = np.array(obj_pos)
        area = np.array(area_pos)
        tcp = np.array(tcp_pos)

        # Direction vector of the line
        line_vec = area - tcp

        # Vector from TCP to the object
        point_vec = obj - tcp

        # Calculate distance between the object and the area
        obj_area_distance = np.linalg.norm(obj - area)

        # Calculate distance between the TCP and the area
        tcp_area_distance = np.linalg.norm(tcp - area)

        # If the TCP is closer to the area than the object, return infinity
        # Object is behind the TCP relative to the area
        # Object is next to the TCP relative to the area
        if tcp_area_distance < obj_area_distance or abs(obj_area_distance - tcp_area_distance) < 0.01:
            return float("inf")

        # Orthogonal distance (cross product of the direction vector with the point vector, normalized by the length of the direction vector)
        distance = np.linalg.norm(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)

        return float(distance)


class DQNAgent:
    def __init__(
        self,
        action_dim,
        epsilon=0.8,
        epsilon_min=0.1,
        epsilon_decay=0.9999,
        supervisor_epsilon=0.5,
        gamma=0.99,
        input_shape=(88, 88, 6),
        weights_path="",
        weights_dir="models/best",
        learning_rate=0.00025,  # Add learning_rate parameter
        use_pretrained_best_model=False,  # Add use_pretrained_best_model parameter
        auto_set_epsilon=True,  # Add auto_set_epsilon parameter
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.supervisor_epsilon = supervisor_epsilon
        self.gamma = gamma
        self.input_shape = input_shape
        self.use_pretrained_best_model = use_pretrained_best_model
        self.auto_set_epsilon = auto_set_epsilon
        self.agent_actions = []  # Store only agent actions for plotting
        self.supervisor_actions = []  # Store only supervisor actions for plotting

        # Set start episode to 0 if no weights are loaded
        self.start_episode = 0

        # Create main and target networks
        self.model = ConvDQN()

        # Build models with dummy input
        dummy_state = np.zeros((1,) + self.input_shape)
        self.model(dummy_state)  # Initialize with correct shape

        # Create and initialize target model
        self.target_model = ConvDQN()
        self.target_model(dummy_state)  # Initialize with correct shape

        if self.use_pretrained_best_model and weights_path:
            try:
                weights_file_path = os.path.join(weights_dir, weights_path)

                # Load the weights from the file
                self.model.load_weights(weights_file_path)
                logger.debug(f"Loaded weights from {weights_file_path}")

                # Extract the episode number from the weights file
                self.start_episode = int(weights_path.split("_")[-1])

                # Calculate the epsilon value based on the episode number and current set of parameters
                if self.auto_set_epsilon:
                    self.epsilon = max(epsilon_min, epsilon * (epsilon_decay ** (self.start_episode * 200))) # 200 is mean steps per episode
                    logger.debug(f"Setting epsilon to {self.epsilon:.2f} based on the start episode number {self.start_episode}")
                else:
                    self.start_episode = 0
                    logger.debug(f"Keeping epsilon at {self.epsilon:.2f} and reset the start episode number to 0")

            except Exception as e:
                logger.error(f"Error loading weights from {weights_file_path}: {e}")
        else:
            logger.debug("Starting model with random weights")

        # Copy the weights from the main model also to the target model
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Use the learning_rate parameter

    def get_action(self, state, supervisor, training=True):
        # Explanation of the epsilon-greedy strategy:
        # With probability epsilon, take a random action (exploration) or ask the supervisor for an action
        # With probability 1 - epsilon, take the action with the highest Q-value (exploitation)

        if training and np.random.random() < self.epsilon:

            if np.random.random() < self.supervisor_epsilon:
                # Ask supervisor for action
                logger.info(f"Supervisor-Action with epsilon {self.epsilon:.2f}")
                action = supervisor.ask_supervisor()
                self.supervisor_actions.append(action)  # Store supervisor actions for plotting
            else:
                logger.info(f"Random action taken with {self.epsilon:.2f} and supervisor epsilon {self.supervisor_epsilon:.2f}")
                action = np.random.uniform(-1, 1, self.action_dim)
                self.supervisor_actions.append(action)

            return action
        
        else:
            logger.info(f"  Agent-Action    with epsilon {self.epsilon:.2f}")
            state = np.expand_dims(state, axis=0)

            # Direct continuous output from network
            heatmap = self.model(state)[0].numpy()  

            action, pixels = self._choose_action_from_max_area(heatmap) # output is vector [x, y] with values between -1 and 1
            logger.debug(f"Action for Agent: {action} and pixels: {pixels}")
            self.agent_actions.append(action)  # Store agent actions for plotting

        # Purge oldest actions if the length exceeds 10500
        if len(self.agent_actions) > 10500:
            self.agent_actions = self.agent_actions[-10500:]
        if len(self.supervisor_actions) > 10500:
            self.supervisor_actions = self.supervisor_actions[-10500:]

        return action

    def train(self, replay_buffer, batch_size=32, train_start_size=1000, beta=0.4, window_size=5):
        # Check if the replay buffer has enough samples to train
        if replay_buffer.size() < batch_size or replay_buffer.size() < train_start_size:
            logger.info(f"Replay buffer size: {replay_buffer.size()} is less than batch size: {batch_size} or train start size: {train_start_size}. Skip training.")
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)

        # Predict heatmaps for the next states using the target model
        target_heatmaps = self.target_model(next_states).numpy()  # (batch_size, H, W, 1)

        # Initialize the list to store the maximum Q-values from the local neighborhood
        next_values = []
        pixel_x_list = []
        pixel_y_list = []

        # Iterate through each heatmap to calculate the maximum Q-value within a local neighborhood of the current action
        # This ignores artefacts and focuses on the most relevant part of the heatmap
        # This explicit method to calculate the maximum Q-value only works because the heatmaps represents to total action space (of the robot, camera bounds and movement bounds need to match)
        for i, heatmap in enumerate(target_heatmaps):
            # Normalize the heatmap
            heatmap_normalized = (heatmap - np.mean(heatmap)) / (np.std(heatmap) + 1e-8)

            # Smooth the heatmap to reduce noise
            heatmap_smoothed = scipy.ndimage.gaussian_filter(heatmap_normalized, sigma=1)

            # Convert action to pixel coordinates
            action_x, action_y = actions[i]
            height, width = heatmap_smoothed.shape[:2]
            pixel_x = int((action_x + 1) * (width - 1) / 2)
            pixel_y = int((action_y + 1) * (height - 1) / 2)
            
            # Append pixel coordinates to the lists
            pixel_x_list.append(pixel_x)
            pixel_y_list.append(pixel_y)

            # Define the local window around the action's pixel
            x_min = max(0, pixel_x - window_size // 2)
            x_max = min(width, pixel_x + window_size // 2 + 1)
            y_min = max(0, pixel_y - window_size // 2)
            y_max = min(height, pixel_y + window_size // 2 + 1)

            # Extract the local neighborhood
            local_heatmap = heatmap_smoothed[y_min:y_max, x_min:x_max]

            # Find the maximum value in the local neighborhood
            max_index_local = np.unravel_index(np.argmax(local_heatmap), local_heatmap.shape)

            # Map the local index back to global coordinates
            global_index = (max_index_local[0] + y_min, max_index_local[1] + x_min)

            # Append the maximum Q-value from the local neighborhood
            next_values.append(heatmap_smoothed[global_index])

        next_values = np.array(next_values).squeeze()  # Convert list to numpy array (batch_size,)

        # Calculate target Q-values
        target_values = rewards + (1 - dones) * next_values * self.gamma

        # Train the model
        with tf.GradientTape() as tape:
            # Predict Q-values for the current states
            values = self.model(states)
            
            # Gather Q-values for the executed actions
            values = tf.gather_nd(values, tf.stack([np.arange(len(values)), pixel_y_list, pixel_x_list], axis=1))

            # Compute the loss
            loss = tf.keras.losses.MSE(target_values, values)
            weighted_loss = weights * loss

            logger.debug(f"Weighted Loss: {weighted_loss.numpy()}")

        # Compute gradients and apply updates
        grads = tape.gradient(weighted_loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update priorities in the replay buffer
        replay_buffer.update_losses(indices, weighted_loss.numpy())

        # Perform epsilon annealing
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return weighted_loss.numpy()

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def _choose_action_from_max_area(self, heatmap, window_size=3):
        # Squeeze the batch and channel dimensions
        heatmap = heatmap.squeeze()  # Remove batch and channel dimension (if any)

        # Find the position with the maximum value in the heatmap
        max_index = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        # Extract the window around the maximum value
        i_min = max(max_index[0] - window_size // 2, 0)
        i_max = min(max_index[0] + window_size // 2 + 1, heatmap.shape[0])
        j_min = max(max_index[1] - window_size // 2, 0)
        j_max = min(max_index[1] + window_size // 2 + 1, heatmap.shape[1])

        # Extract the local area around the maximum value
        local_area = heatmap[i_min:i_max, j_min:j_max]

        # Find the maximum value in the local area
        local_max_index = np.unravel_index(np.argmax(local_area), local_area.shape)

        # Calculate the global index of the maximum value
        global_index = (local_max_index[0] + i_min, local_max_index[1] + j_min)        

        # Normalize the (i, j) position to the range [-1, 1]
        height, width = heatmap.shape
        normalized_x = 2 * global_index[0] / (height - 1) - 1
        normalized_y = 2 * global_index[1] / (width - 1) - 1

        # DEBUG: Convert heatmap to rgb, resize it to 500x500, make max value 255 and min value 0 and display it with opencv
        # Normalize the heatmap to the range [0, 1]
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

        # Resize heatmap to 500x500
        heatmap_resized = cv2.resize(heatmap, (500, 500), interpolation=cv2.INTER_NEAREST)

        # Copy heatmap
        copy_heatmap = copy.deepcopy(heatmap_resized)

        # Change
        copy_heatmap[global_index[0],global_index[1]] = 1

        # Display the heatmap with OpenCV
        cv2.imshow("Original Grayscale Heatmap", copy_heatmap)
        cv2.waitKey(1)

        # Convert heatmap to RGB
        heatmap_rgb = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Get max value of heatmap and min value of heatmap
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        min_pos = np.unravel_index(np.argmin(heatmap), heatmap.shape)

        # Make pixel at max_value red and pixel at min_value blue
        heatmap_rgb[max_pos[0], max_pos[1], :] = [255, 255, 255]  # White
        heatmap_rgb[min_pos[0], min_pos[1], :] = [0, 0, 0]  # Black

        # Resize heatmap to 500x500
        heatmap_rgb_resized = cv2.resize(heatmap_rgb, (500, 500), interpolation=cv2.INTER_NEAREST)

        # Display the heatmap with OpenCV
        cv2.imshow("RGB Heatmap", heatmap_rgb_resized)
        cv2.waitKey(1)        

        return np.array([normalized_x, normalized_y]), global_index


def plot_actionHistory(agent_actions, supervisor_actions, plot_dir, episode):
    """Plot agent and supervisor actions with fading colors and save the plot."""
    fig, ax = plt.subplots()
    num_agent_actions = len(agent_actions)
    num_supervisor_actions = len(supervisor_actions)
    agent_colors = plt.cm.Blues(np.linspace(0.3, 1, num_agent_actions))
    supervisor_colors = plt.cm.Greens(np.linspace(0.3, 1, num_supervisor_actions))

    for i, action in enumerate(agent_actions):
        ax.scatter(action[0], action[1], color=agent_colors[i], s=10, label="Agent Action" if i == 0 else "")

    for i, action in enumerate(supervisor_actions):
        ax.scatter(action[0], action[1], color=supervisor_colors[i], s=10, label="Supervisor Action" if i == 0 else "")

    ax.set_xlabel("Action X")
    ax.set_ylabel("Action Y")
    ax.set_title("Agent (blue) and Supervisor (green) Actions Over Time")
    ax.legend()

    # Save the plot image
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/agent_supervisor_actions_{episode}.png")
    plt.close()


def plot_rewards_epsilons(rewards, epsilons, episode, plot_dir):
    # Create the figure and the first y-axis
    fig, ax1 = plt.subplots()

    # Plot the rewards on the first y-axis
    ax1.plot(rewards, label="Reward", color="tab:blue")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Create the second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the epsilons on the second y-axis
    ax2.plot(epsilons, label="Epsilon", color="tab:orange")
    ax2.set_ylabel("Epsilon", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Add the legend
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # test if plot_dir exists, if not create it
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot image
    plt.tight_layout()  # Avoid cutting off labels
    plt.savefig(f"{plot_dir}/dqn_rewards_epsilons_{episode}.png")
    plt.close()

    # Save the rewards and epsilons to a CSV file
    data = np.column_stack((rewards, epsilons))
    np.savetxt(f"{plot_dir}/dqn_rewards_epsilons_{episode}.csv", data, delimiter=",", header="Reward,Epsilon", comments="")


@hydra.main(version_base=None, config_path="config", config_name="DQN")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    # Setup environment - copy from play_game.py
    bullet_client = setup_bullet_client(cfg.render)
    robot = instantiate(cfg.robot, bullet_client=bullet_client)

    # Create task factory and other components first
    t_bounds = copy.deepcopy(robot.workspace_bounds)
    t_bounds[2, 1] = t_bounds[2, 0]
    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    t_center = np.mean(t_bounds, axis=1)
    teletentric_camera = instantiate(cfg.teletentric_camera, bullet_client=bullet_client, t_center=t_center, robot=robot)

    # Create environment with all components
    env = PushingEnv(
        debug=cfg.debug,
        bullet_client=bullet_client,
        robot=robot,
        task_factory=task_factory,
        teletentric_camera=teletentric_camera,
        workspace_bounds=cfg.workspace_bounds,
        movement_bounds=cfg.movement_bounds,
        step_size=cfg.step_size,
        gripper_offset=cfg.gripper_offset,
        fixed_z_height=cfg.fixed_z_height,
        absolut_movement=cfg.absolut_movement,
        distance_TCP_obj_reward_scale=cfg.distance_TCP_obj_reward_scale,
        distance_obj_area_reward_scale=cfg.distance_obj_area_reward_scale,
        iou_reward_scale=cfg.iou_reward_scale,  # Pass the parameter
        no_movement_threshold=cfg.no_movement_threshold,
        max_moves_without_positive_reward=cfg.max_moves_without_positive_reward,
        success_threshold_trans=cfg.success_threshold_trans,
        success_threshold_rot=cfg.success_threshold_rot,
        activate_distance_obj_area_reward=cfg.activate_distance_obj_area_reward,
        activate_distance_TCP_obj_reward=cfg.activate_distance_TCP_obj_reward,
        activate_iou_reward=cfg.activate_iou_reward,
        activate_moves_without_positive_reward=cfg.activate_moves_without_positive_reward,
        activate_no_movement_punishment=cfg.activate_no_movement_punishment,
        angle_obj_area_tcp_threshold=cfg.angle_obj_area_tcp_threshold,
    )

    logger.info("Instantiation completed.")
    logger.info("Starting training with following activated rewards:")
    logger.info(f"Distance TCP-Object Reward: {cfg.activate_distance_TCP_obj_reward}")
    logger.info(f"Distance Object-Area Reward: {cfg.activate_distance_obj_area_reward}")
    logger.info(f"IoU Reward: {cfg.activate_iou_reward}")
    logger.info(f"Moves without positive reward Reward: {cfg.activate_moves_without_positive_reward}")
    logger.info(f"No movement punishment Reward: {cfg.activate_no_movement_punishment}")

    # Initialize DQN agent with 2D continuous action space
    action_dim = 2  # (x,y) continuous actions
    input_shape = (88, 88, 6)  # RGB (3) + 3 * depth (1) = 6  channels, 88x88 pixels size needed for ResNet (divisible by 8)
    supervisor = DQNSupervisor(
        action_dim,
        env,
        workspace_bounds=cfg.workspace_bounds,
        sv_90deg_movement_threshold=cfg.supervisor.sv_90deg_movement_threshold
    )
    agent = DQNAgent(
        action_dim,
        input_shape=input_shape,
        weights_path=cfg.weights_path,
        weights_dir=cfg.weights_dir,
        learning_rate=cfg.learning_rate,  # Pass the learning_rate from the config
        use_pretrained_best_model=cfg.use_pretrained_best_model,  # Pass the use_pretrained_best_model from the config
        auto_set_epsilon=cfg.auto_set_epsilon,  # Pass the auto_set_epsilon from the config
    )
    logger.info("DQN supervisor and DQN agent initialized.")
    replay_buffer = PrioritizedReplayBuffer()
    logger.info("Replay buffer initialized.")

    # Initialize reward tracking
    rewards = []
    epsilons = []

    # Training loop
    for episode in range(cfg.num_episodes):
        state = env.reset()
        episode_reward = 0

        # Adjust max steps per episode for the first few episodes to improve learning speed
        if cfg.weights_path:  # if pretrained model is loaded, use max steps from config
            max_steps = cfg.max_steps_per_episode
        else:
            max_steps = min(cfg.max_steps_per_episode, (episode + 1) * 10)
        logger.debug(f"Starting episode {episode} with max steps {max_steps}.")

        for step in range(max_steps):
            action = agent.get_action(state, supervisor)

            # Get next state using environment's step function
            next_state, reward, done, _, failed = env.step(action)

            replay_buffer.put(state, action, reward, next_state, done)

            if replay_buffer.size() >= cfg.batch_size:
                loss = agent.train(replay_buffer, cfg.batch_size)

            if step % cfg.target_update_freq == 0:
                agent.update_target()

            state = next_state
            episode_reward += reward

            if done:
                logger.debug(f"Episode {episode} completed at step {step}. Reward = {episode_reward}")
                logger.error(f"Episode {episode} completed at step {step}. Reward = {episode_reward}")
                break

            if failed:
                logger.debug(f"Episode {episode} failed at step {step} because all objects are outside of workspace.")
                logger.error(f"Episode {episode} failed at step {step} because all objects are outside of workspace.")
                break

        logger.debug(f"Episode {episode}: Reward = {episode_reward}")
        logger.error(f"Episode {episode}: Reward = {episode_reward}")

        # Save rewards and epsilon for plotting
        rewards.append(episode_reward)
        epsilons.append(agent.epsilon)

        # Plot rewards and epsilon in the same graph and save in to file periodically
        if episode % cfg.plot_freq == 0 and episode > 0:
            plot_rewards_epsilons(rewards, epsilons, episode, cfg.plot_dir)
            #plot_actionHistory(agent.agent_actions, agent.supervisor_actions, cfg.plot_dir, episode)  # Plot agent and supervisor actions

        # Save model periodically
        if episode % cfg.save_freq == 0 and episode > 0:
            agent.model.save_weights(f"{cfg.model_dir}/dqn_episode_{episode}", save_format="tf")

    env.close()
    logger.debug("Training completed.")
    logger.error("Training completed.")


if __name__ == "__main__":
    main()
