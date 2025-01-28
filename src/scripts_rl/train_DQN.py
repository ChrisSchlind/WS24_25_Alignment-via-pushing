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
import scipy.ndimage


class ConvDQN(tf.keras.Model):
    def __init__(self, action_dim=4):
        super().__init__()
        # Increased Conv-Layers to balance VRAM usage and performance
        self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=1, activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(128, 3, strides=1, activation="relu")

        # Flatten the output of the last convolutional layer
        self.flatten = tf.keras.layers.Flatten()

        # Increased Fully Connected Layers
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.fc2 = tf.keras.layers.Dense(128, activation="relu")
        self.fc3 = tf.keras.layers.Dense(64, activation="relu")

        # Output layer for classification with action_dim classes and softmax activation
        self.output_layer = tf.keras.layers.Dense(action_dim, activation="softmax")

    def call(self, x):
        # x: (batch_size, height, width, channels)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the output of the last convolutional layer
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # Output layer for classification
        x = self.output_layer(x)
        logger.debug(f"Output of the network: {x}")

        # Convert vector to index of the maximum value
        x = tf.argmax(x, axis=-1)
        logger.debug(f"Output of the network after argmax: {x}")

        # Squeeze the output to remove the batch dimension
        x = tf.squeeze(x)

        return x  # Output is the index of the maximum value in the output vector

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

class DQNAgent:
    def __init__(
        self,
        action_dim,
        epsilon=0.8,
        epsilon_min=0.1,
        epsilon_decay=0.99995,
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
        self.gamma = gamma
        self.input_shape = input_shape
        self.use_pretrained_best_model = use_pretrained_best_model
        self.auto_set_epsilon = auto_set_epsilon
        self.agent_actions = []  # Store only agent actions for plotting

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
        # With probability 1 - epsilon, take the action with the highest Q-value (exploitation)

        if training and np.random.random() < self.epsilon:
            logger.info(f"Random action taken with epsilon {self.epsilon:.2f}")
            action = np.random.choice([0, 1, 2, 3])
            self.agent_actions.append(action)
            return action
        
        else:
            logger.info(f"Agent-Action with epsilon {self.epsilon:.2f}")
            state = np.expand_dims(state, axis=0)

            # Direct continuous output from network
            heatmap = self.model(state)[0].numpy()  

            action, pixels = self._choose_action_from_min_area(heatmap) # output is vector [x, y] with values between -1 and 1
            logger.debug(f"Action for Agent: {action} and pixels: {pixels}")
            self.agent_actions.append(action)  # Store agent actions for plotting

        # Purge oldest actions if the length exceeds 10500
        if len(self.agent_actions) > 10500:
            self.agent_actions = self.agent_actions[-10500:]

        return action

    def train(self, replay_buffer, batch_size=32, train_start_size=10000, beta=0.4):
        # Check if the replay buffer has enough samples to train
        if replay_buffer.size() < batch_size or replay_buffer.size() < train_start_size:
            logger.info(f"Replay buffer size: {replay_buffer.size()} is less than batch size: {batch_size} or train start size: {train_start_size}. Skip training.")
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)

        # Predict Q-values for the next states using the target model
        next_q_values = self.target_model(next_states).numpy()  # (batch_size, action_dim)

        # Calculate the maximum Q-value for the next states
        max_next_q_values = np.max(next_q_values, axis=1)

        # Calculate target Q-values
        target_q_values = rewards + (1 - dones) * max_next_q_values * self.gamma

        # Train the model
        with tf.GradientTape() as tape:
            # Predict Q-values for the current states
            q_values = self.model(states)

            # Gather Q-values for the executed actions
            action_indices = np.array(actions, dtype=np.int32)
            q_values = tf.gather(q_values, action_indices, batch_dims=1)

            # Compute the loss
            loss = tf.keras.losses.MSE(target_q_values, q_values)
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
        

def plot_actionHistory(agent_actions, plot_dir, episode):
    """Plot agent actions with fading colors and save the plot."""
    fig, ax = plt.subplots()
    num_agent_actions = len(agent_actions)
    agent_colors = plt.cm.Blues(np.linspace(0.3, 1, num_agent_actions))

    for i, action in enumerate(agent_actions):
        ax.scatter(action[0], action[1], color=agent_colors[i], s=10, label="Agent Action" if i == 0 else "")

    ax.set_xlabel("Action X")
    ax.set_ylabel("Action Y")
    ax.set_title("Agent Actions Over Time")
    ax.legend()

    # Save the plot image
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/agent_actions_{episode}.png")
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
        activate_objects_outside_workspace_punishment=cfg.activate_objects_outside_workspace_punishment,
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
    input_shape = (88, 88, 6)  # RGB (3) + 3 * depth (1) = 6  channels
    agent = DQNAgent(
        action_dim,
        input_shape=input_shape,
        weights_path=cfg.weights_path,
        weights_dir=cfg.weights_dir,
        learning_rate=cfg.learning_rate,  # Pass the learning_rate from the config
        use_pretrained_best_model=cfg.use_pretrained_best_model,  # Pass the use_pretrained_best_model from the config
        auto_set_epsilon=cfg.auto_set_epsilon,  # Pass the auto_set_epsilon from the config
    )
    logger.info("DQN agent initialized.")
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
            action = agent.get_action(state)

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
            #plot_actionHistory(agent.agent_actions, cfg.plot_dir, episode)  # Plot agent actions

        # Save model periodically
        if episode % cfg.save_freq == 0 and episode > 0:
            agent.model.save_weights(f"{cfg.model_dir}/dqn_episode_{episode}", save_format="tf")

    env.close()
    logger.debug("Training completed.")
    logger.error("Training completed.")


if __name__ == "__main__":
    main()
