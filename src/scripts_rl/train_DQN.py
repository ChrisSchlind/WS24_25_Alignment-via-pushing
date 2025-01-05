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
from matplotlib import pyplot as plt

from bullet_env.util import setup_bullet_client, stdout_redirected
from transform.affine import Affine
from bullet_env.ur10_cell import UR10Cell  # Import UR10Cell
from bullet_env.pushing_env import PushingEnv  # Add this import


class ConvDQN(tf.keras.Model):
    def __init__(self, action_dim=2):
        super().__init__()
        # Example CNN stack (adjust as needed):
        self.conv1 = tf.keras.layers.Conv2D(16, 3, strides=2, activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(action_dim)  # final layer with no activation
        # We'll apply tanh in call()

    def call(self, x):
        # x: (batch_size, height, width, channels)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        # Output in [-1,1]
        return tf.nn.tanh(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size, reward_range=(-1.0, 1.0)):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*sample))

        # Normalization of rewards
        # Find the minimum and maximum of the rewards
        reward_min = np.min(rewards)
        reward_max = np.max(rewards)

        # Normalize the rewards to the desired range
        # Formula: norm_reward = (reward - min) / (max - min) * (range_max - range_min) + range_min
        reward_min_new, reward_max_new = reward_range
        rewards_normalized = (rewards - reward_min) / (reward_max - reward_min) * (reward_max_new - reward_min_new) + reward_min_new

        return states, actions, rewards_normalized, next_states, dones

    def size(self):
        return len(self.buffer)
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        # Standard replay buffer
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # List of priorities
        self.alpha = alpha  # Parameter that controls the priority scale

    def _get_priority(self, error):
        # Priority is calculated based on the TD error
        return float((error + 1e-5) ** self.alpha)  # Avoid division by zero

    def put(self, state, action, reward, next_state, done, error=1.0):
        # Add experience with an initial priority
        priority = self._get_priority(error / (1 + np.abs(error)))

        # Increase the priority if the reward is positive
        if reward > 0:
            priority *= 1.5

        self.buffer.append([state, action, reward, next_state, done])
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4, reward_range=(-1.0, 1.0)):
        # Calculate the total priority of all experiences
        priorities = np.array(self.priorities)
        prob_dist = priorities / np.sum(priorities)

        # Select experiences based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=prob_dist)

        # Calculate Importance Sampling Weights
        weights = (len(self.buffer) * prob_dist[indices]) ** (-beta)
        weights /= weights.max()  # Normalize the weights

        # Retrieve the sampled experiences and their priorities
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

        # Normalization of rewards
        # Find the minimum and maximum of the rewards
        reward_min = np.min(rewards)
        reward_max = np.max(rewards)

        # Normalize the rewards to the desired range
        # Formula: norm_reward = (reward - min) / (max - min) * (range_max - range_min) + range_min
        reward_min_new, reward_max_new = reward_range
        rewards_normalized = (rewards - reward_min) / (reward_max - reward_min) * (reward_max_new - reward_min_new) + reward_min_new

        logger.debug(f"Rewards: {rewards}")
        logger.debug(f"Normalized Rewards: {rewards_normalized}")

        return states, actions, rewards_normalized, next_states, dones, indices, weights

    def update_priorities(self, indices, errors):
        # Update the priorities for the given indices based on the errors
        for idx, error in zip(indices, errors):
            priority = self._get_priority(error[1] / (1 + np.abs(error[1])))
            self.priorities[idx] = priority

    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, action_dim, epsilon=0.8, epsilon_min=0.1, epsilon_decay=0.9995, gamma=0.99, input_shape=(84, 84, 4)):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.input_shape = input_shape

        # Create main and target networks
        self.model = ConvDQN(action_dim)

        # Build models with dummy input
        dummy_state = np.zeros((1,) + input_shape)
        self.model(dummy_state)  # Initialize with correct shape

        # Create and initialize target model
        self.target_model = ConvDQN(action_dim)
        self.target_model(dummy_state)  # Initialize with correct shape

        # Now we can safely set weights
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)

    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            # Random action in continuous space
            logger.debug(f"Random action taken with {self.epsilon}")
            return np.random.uniform(-1, 1, self.action_dim)

        state = np.expand_dims(state, axis=0)
        # Direct continuous output from network
        action = self.model(state)[0].numpy()

        # Ensure actions are in [-1,1] range
        return np.clip(action, -1, 1)

    def train(self, replay_buffer, batch_size=32):
        # Check if the replay buffer is of the correct type
        if not isinstance(replay_buffer, ReplayBuffer):
            raise TypeError("The replay buffer used must be an instance of ReplayBuffer. Change replay_buffer in main from PrioritizedReplayBuffer to ReplayBuffer or use train_prioritized method.")

        if replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, reward_range=(-1.0, 1.0))

        targets = self.target_model(states).numpy()
        next_value = np.max(self.target_model(next_states).numpy(), axis=1)
        action_indices = np.argmax(actions, axis=1)
        targets[range(actions.shape[0]), action_indices] = rewards + (1 - dones) * next_value * self.gamma

        with tf.GradientTape() as tape:
            values = self.model(states)
            # MSE loss for continuous actions
            loss = tf.keras.losses.MSE(targets, values)  # changed actions - predicted_actions

        logger.debug(f"Loss: {loss}")

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Epsilon Annealing: Reduce epsilon after each training step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.numpy()
    
    def train_prioritized(self, replay_buffer, batch_size=32, beta=0.4):
        # Check if the replay buffer is of the correct type
        if not isinstance(replay_buffer, PrioritizedReplayBuffer):
            raise TypeError("The replay buffer used must be an instance of PrioritizedReplayBuffer. Change replay_buffer in main from ReplayBuffer to PrioritizedReplayBuffer or use train method.")

        if replay_buffer.size() < batch_size:
            return

        # Sample a batch from the prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta, reward_range=(-1.0, 1.0))

        #logger.debug(f"Actions: {actions}")
        #logger.debug(f"Rewards: {rewards}")

        # Calculate target values for Q-learning
        targets = self.target_model(states).numpy()
        next_value = np.max(self.target_model(next_states).numpy(), axis=1)
        action_indices = np.argmax(actions, axis=1)
        targets[range(actions.shape[0]), action_indices] = rewards + (1 - dones) * next_value * self.gamma

        # Ensure no NaN values in targets or values
        targets = tf.debugging.check_numerics(targets, message="targets contains NaN or Inf")

        # Calculate predictions
        with tf.GradientTape() as tape:
            values = self.model(states)

            # Ensure no NaN values in predictions
            values = tf.debugging.check_numerics(values, message="values contains NaN or Inf")

            # Ensure that values are not None or NaN
            if np.any(np.isnan(values.numpy())) or np.any(np.isnan(targets)):
                raise ValueError("NaN detected in predictions or targets.")

            # Debugging the loss computation
            loss = tf.keras.losses.MSE(targets, values)  # Use manual MSE computation

        # Calculate the error (difference between predicted value and target value)
        errors = np.abs(targets - values.numpy())

        # Apply the Importance Sampling Weights
        weighted_loss = weights * loss  # TensorFlow computation, maintaining the gradient flow

        logger.debug(f"Weights: {weights}")
        logger.debug(f"Weighted Loss: {weighted_loss}")

        # Check if any gradients are None
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Debugging the gradients
        if any(grad is None for grad in grads):
            logger.error(f"Gradients are None for the following layers: {[var.name for var, grad in zip(self.model.trainable_variables, grads) if grad is None]}")
            raise ValueError("One or more gradients are None!")

        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update priorities in the replay buffer
        replay_buffer.update_priorities(indices, errors)

        # Epsilon Annealing: Reduce epsilon after each training step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return weighted_loss.numpy()

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())


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
        distance_reward_scale=cfg.distance_reward_scale,
        iou_reward_scale=cfg.iou_reward_scale,  # Pass the parameter
        no_movement_threshold=cfg.no_movement_threshold,
    )

    logger.info("Instantiation completed.")

    # Initialize DQN agent with 2D continuous action space
    action_dim = 2  # (x,y) continuous actions
    input_shape = (84, 84, 4)  # RGB (3) + depth (1) = 4 channels
    agent = DQNAgent(action_dim, input_shape=input_shape)
    logger.info("DQN agent initialized.")
    replay_buffer = PrioritizedReplayBuffer() #PrioritizedReplayBuffer() #ReplayBuffer()
    logger.info("Replay buffer initialized.")

    # Initialize reward tracking
    rewards = []
    epsilons = []

    # Training loop
    for episode in range(cfg.num_episodes):
        state = env.reset()
        episode_reward = 0

        # Adjust max steps per episode for the first few episodes to improve learning speed
        max_steps = min(cfg.max_steps_per_episode, (episode + 1) * 10)
        logger.debug(f"Starting episode {episode} with max steps {max_steps}.")

        for step in range(max_steps):
            action = agent.get_action(state)

            # Get next state using environment's step function
            next_state, reward, done, _ = env.step(action)

            replay_buffer.put(state, action, reward, next_state, done)

            if replay_buffer.size() >= cfg.batch_size:
                loss = agent.train_prioritized(replay_buffer, cfg.batch_size)

            if step % cfg.target_update_freq == 0:
                agent.update_target()

            state = next_state
            episode_reward += reward

            if done:
                break

        logger.debug(f"Episode {episode}: Reward = {episode_reward}")

        # Save rewards and epsilon for plotting
        rewards.append(episode_reward)
        epsilons.append(agent.epsilon)

        # Plot rewards and epsilon in the same graph and save in to file periodically
        if episode % cfg.plot_freq == 0 and episode > 0:
            plot_rewards_epsilons(rewards, epsilons, episode, cfg.plot_dir)

        # Save model periodically
        if episode % cfg.save_freq == 0 and episode > 0:
            agent.model.save_weights(f"{cfg.model_dir}/dqn_episode_{episode}")

    env.close()
    logger.debug("Training completed.")

def plot_rewards_epsilons(rewards, epsilons, episode, plot_dir):
    # Create the figure and the first y-axis
    fig, ax1 = plt.subplots()

    # Plot the rewards on the first y-axis
    ax1.plot(rewards, label="Reward", color="tab:blue")
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # Create the second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the epsilons on the second y-axis
    ax2.plot(epsilons, label="Epsilon", color="tab:orange")
    ax2.set_ylabel('Epsilon', color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    # Add the legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot image
    plt.tight_layout()  # Avoid cutting off labels
    plt.savefig(f"{plot_dir}/dqn_rewards_epsilons_{episode}.png")
    plt.close()


if __name__ == "__main__":
    main()
