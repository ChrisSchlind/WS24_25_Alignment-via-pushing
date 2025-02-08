import numpy as np
import os
import tensorflow as tf
from collections import deque
import logging
import random

logger = logging.getLogger(__name__)

class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6):
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
        epsilon_decay=0.99995,
        supervisor_epsilon=0.0,
        gamma=0.99,
        input_shape=(84, 84, 6),
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
        logger.debug(f"Model initialized with input shape: {self.input_shape}")

        # Create and initialize target model
        self.target_model = ConvDQN()
        self.target_model(dummy_state)  # Initialize with correct shape
        logger.debug("Target model initialized")

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

    def get_action(self, state, training=True):
        # Explanation of the epsilon-greedy strategy:
        # With probability epsilon, take a random action (exploration) 

        if training and np.random.random() < self.epsilon:
            logger.info(f"Random action taken with epsilon {self.epsilon:.2f}")
            action = np.zeros(self.action_dim)
            action[np.random.choice(self.action_dim)] = 1
            self.agent_actions.append(action)
            return action
        
        else:
            logger.info(f"Agent-Action with epsilon {self.epsilon:.2f}")

            # Add batch dimension to the state
            state = np.expand_dims(state, axis=0)

            # Direct continuous output from network
            action = self.model(state)[0].numpy()
            logger.debug(f"Action for Agent: {action}")

            self.agent_actions.append(action)  # Store agent actions for plotting

        # Purge oldest actions if the length exceeds 10500
        if len(self.agent_actions) > 10500:
            self.agent_actions = self.agent_actions[-10500:]

        return action

    def train(self, replay_buffer, batch_size=32, train_start_size=4000, beta=0.4):
        # Check if the replay buffer has enough samples to train
        if replay_buffer.size() < batch_size or replay_buffer.size() < train_start_size:
            logger.info(f"Replay buffer size: {replay_buffer.size()} is less than batch size: {batch_size} or train start size: {train_start_size}. Skip training.")
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)

        # Compute the target Q-values
        targets = self.target_model(states).numpy()

        # Compute the next Q-values using the target model
        next_value = self.target_model(next_states).numpy().max(axis=1)

        # new approximation of the action value; the '1-dones' means, that if the game
        targets[range(actions.shape[0]), np.argmax(actions, axis=1)] = rewards + (1 - dones) * next_value * self.gamma

        # Train the model
        with tf.GradientTape() as tape:
            # Predict values for the current states
            values = self.model(states)

            # Compute the loss
            loss = tf.keras.losses.MSE(targets, values)
            logger.debug(f"Loss: {loss.numpy()}")
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

        return loss.numpy()

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())
    