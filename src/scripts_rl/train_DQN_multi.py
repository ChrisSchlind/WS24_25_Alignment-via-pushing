import sys
import copy
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import random
from collections import deque
from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate
import hydra
from bullet_env.util import setup_bullet_client
from bullet_env.pushing_env import PushingEnv


class ConvDQN(tf.keras.Model):
    def __init__(self, action_dim=2):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, 3, strides=2, activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation="tanh")
        self.fc2 = tf.keras.layers.Dense(action_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return tf.nn.tanh(x)


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

        # Initialize the rewards
        reward_min = 0
        reward_max = 0

        while (reward_min == 0 and reward_max == 0) or reward_min == reward_max: # Avoid division by zero, sample again
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

            if (reward_min == 0 and reward_max == 0) or reward_min == reward_max:
                logger.debug("Resampling due to zero rewards.")

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
    def __init__(self, action_dim, epsilon=0.8, epsilon_min=0.1, epsilon_decay=0.9999, gamma=0.99, input_shape=(84, 84, 4)):
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
    
    def train(self, replay_buffer, batch_size=32, beta=0.4):
        # Check if the replay buffer is of the correct type
        if not isinstance(replay_buffer, PrioritizedReplayBuffer):
            raise TypeError("The replay buffer used must be an instance of PrioritizedReplayBuffer. Change replay_buffer in main from ReplayBuffer to PrioritizedReplayBuffer or use train method.")

        if replay_buffer.size() < batch_size:
            return

        # Sample a batch from the prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta, reward_range=(-1.0, 1.0))

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


def worker(env_config, model_weights_queue, replay_queue, cfg, agent):
    bullet_client = setup_bullet_client(env_config.render)
    robot = instantiate(env_config.robot, bullet_client=bullet_client)
    # Create task factory and other components first
    t_bounds = copy.deepcopy(robot.workspace_bounds)
    t_bounds[2, 1] = t_bounds[2, 0]
    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    t_center = np.mean(t_bounds, axis=1)
    teletentric_camera = instantiate(cfg.teletentric_camera, bullet_client=bullet_client, t_center=t_center, robot=robot)

    env = PushingEnv(
        debug=env_config.debug,
        bullet_client=bullet_client,
        robot=robot,
        task_factory=task_factory,
        teletentric_camera=teletentric_camera,
        workspace_bounds=env_config.workspace_bounds,
        movement_bounds=cfg.movement_bounds,
        step_size=cfg.step_size,
        gripper_offset=cfg.gripper_offset,
        fixed_z_height=cfg.fixed_z_height,
        absolut_movement=cfg.absolut_movement,
        distance_reward_scale=cfg.distance_reward_scale,
        iou_reward_scale=cfg.iou_reward_scale,  # Pass the parameter
    )

    state = env.reset()
    # Make a dummy pass through the model to initialize its weights
    dummy_state = np.zeros((1,) + state.shape)  # Assuming state has shape (84, 84, 4)
    agent.model(dummy_state)  # This initializes the weights

    while True:
        if not model_weights_queue.empty():
            model_weights = model_weights_queue.get()
            agent.model.set_weights(model_weights)  # Apply the model weights

        action = agent.get_action(state)  # Use the agent to select an action
        next_state, reward, done, _ = env.step(action)
        replay_queue.put((state, action, reward, next_state, done))

        if done:
            state = env.reset()
            logger.debug("#################################################################################################")
            logger.info("Episode done, resetting environment")
            logger.debug("#################################################################################################")
        else:
            state = next_state

def train_model(env_config, num_envs):
    model_weights_queue = mp.Queue(maxsize=10)
    replay_queue = mp.Queue(maxsize=10000)
    replay_buffer = PrioritizedReplayBuffer()

    agent = DQNAgent(action_dim=2, input_shape=(84, 84, 4))  # Create the agent here

    processes = []
    for _ in range(num_envs):
        p = mp.Process(target=worker, args=(env_config, model_weights_queue, replay_queue, agent))
        p.start()
        processes.append(p)

    for episode in range(env_config.num_episodes):
        while not replay_queue.empty():
            replay_buffer.put(*replay_queue.get())

        if replay_buffer.size() >= env_config.batch_size:
            loss = agent.train(replay_buffer, env_config.batch_size)

        if episode % env_config.target_update_freq == 0:
            agent.update_target()

        if episode % env_config.save_freq == 0:
            agent.model.save_weights(f"{env_config.model_dir}/dqn_episode_{episode}")

        # Send updated weights to workers
        model_weights_queue.put(agent.model.get_weights())

    for p in processes:
        p.terminate()
        p.join()


@hydra.main(version_base=None, config_path="config", config_name="DQN")
def main(cfg: DictConfig):
    train_model(cfg, num_envs=cfg.num_envs)


if __name__ == "__main__":
    main()
