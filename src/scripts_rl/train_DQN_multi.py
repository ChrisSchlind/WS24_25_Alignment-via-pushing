import sys
import copy
import multiprocessing as mp
import numpy as np
import tensorflow as tf
import random
import os
from collections import deque
from loguru import logger
from omegaconf import DictConfig
from hydra.utils import instantiate
import hydra
from bullet_env.util import setup_bullet_client
from bullet_env.pushing_env import PushingEnv

# Initialize CUDA context
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define the Convolutional Deep Q-Network (DQN) model
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

# Define the Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    # Calculate priority based on error
    def _get_priority(self, error):
        return float((error + 1e-5) ** self.alpha)

    # Add experience to the buffer
    def put(self, state, action, reward, next_state, done, error=1.0):
        priority = self._get_priority(error / (1 + np.abs(error)))
        if reward > 0:
            priority *= 1.5
        self.buffer.append([state, action, reward, next_state, done])
        self.priorities.append(priority)

    # Sample a batch of experiences from the buffer
    def sample(self, batch_size, beta=0.4, reward_range=(-1.0, 1.0)):
        priorities = np.array(self.priorities)
        prob_dist = priorities / np.sum(priorities)
        reward_min = 0
        reward_max = 0

        while (reward_min == 0 and reward_max == 0) or reward_min == reward_max:
            indices = np.random.choice(len(self.buffer), batch_size, p=prob_dist)
            weights = (len(self.buffer) * prob_dist[indices]) ** (-beta)
            weights /= weights.max()

            samples = [self.buffer[idx] for idx in indices]
            states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

            reward_min = np.min(rewards)
            reward_max = np.max(rewards)

            if (reward_min == 0 and reward_max == 0) or reward_min == reward_max:
                logger.debug("Resampling due to zero rewards.")

        reward_min_new, reward_max_new = reward_range
        rewards_normalized = (rewards - reward_min) / (reward_max - reward_min) * (reward_max_new - reward_min_new) + reward_min_new

        logger.debug(f"Rewards: {rewards}")
        logger.debug(f"Normalized Rewards: {rewards_normalized}")

        return states, actions, rewards_normalized, next_states, dones, indices, weights

    # Update priorities of sampled experiences
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = self._get_priority(error[1] / (1 + np.abs(error[1])))
            self.priorities[idx] = priority

    # Get the size of the buffer
    def size(self):
        return len(self.buffer)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, action_dim, epsilon=0.8, epsilon_min=0.1, epsilon_decay=0.9999, gamma=0.99, input_shape=(84, 84, 4), weights_path="", weights_dir="models/best"):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.input_shape = input_shape

        self.model = ConvDQN(action_dim)
        dummy_state = np.zeros((1,) + input_shape)
        self.model(dummy_state)

        self.target_model = ConvDQN(action_dim)
        self.target_model(dummy_state)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)

        if weights_path:
            try:
                weights_file_path = os.path.join(weights_dir, weights_path)
                logger.debug(f"Final weights path: {weights_file_path}")
                self.model.load_weights(weights_file_path)
                logger.debug(f"Loaded weights from {weights_file_path}")
            except Exception as e:
                logger.error(f"Error loading weights from {weights_file_path}: {e}")
        else:
            logger.debug("Starting model with random weights")

        self.target_model.set_weights(self.model.get_weights())

    # Get action based on the current state
    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            logger.debug(f"Random action taken with {self.epsilon}")
            return np.random.uniform(-1, 1, self.action_dim)

        state = np.expand_dims(state, axis=0)
        action = self.model(state)[0].numpy()
        return np.clip(action, -1, 1)
    
    # Train the agent using experiences from the replay buffer
    def train(self, replay_buffer, batch_size=32, beta=0.4):
        if not isinstance(replay_buffer, PrioritizedReplayBuffer):
            raise TypeError("The replay buffer used must be an instance of PrioritizedReplayBuffer. Change replay_buffer in main from ReplayBuffer to PrioritizedReplayBuffer or use train method.")

        if replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta, reward_range=(-1.0, 1.0))

        targets = self.target_model(states).numpy()
        next_value = np.max(self.target_model(next_states).numpy(), axis=1)
        action_indices = np.argmax(actions, axis=1)
        targets[range(actions.shape[0]), action_indices] = rewards + (1 - dones) * next_value * self.gamma

        targets = tf.debugging.check_numerics(targets, message="targets contains NaN or Inf")

        with tf.GradientTape() as tape:
            values = self.model(states)
            values = tf.debugging.check_numerics(values, message="values contains NaN or Inf")
            if np.any(np.isnan(values.numpy())) or np.any(np.isnan(targets)):
                raise ValueError("NaN detected in predictions or targets.")
            loss = tf.keras.losses.MSE(targets, values)

        errors = np.abs(targets - values.numpy())

        weighted_loss = weights * loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        if any(grad is None for grad in grads):
            logger.error(f"Gradients are None for the following layers: {[var.name for var, grad in zip(self.model.trainable_variables, grads) if grad is None]}")
            raise ValueError("One or more gradients are None!")

        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        replay_buffer.update_priorities(indices, errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return weighted_loss.numpy()

    # Update the target model with weights from the main model
    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

# Worker function to run the environment and collect experiences
def worker(cfg, model_weights_queue, replay_queue, agent):
    bullet_client = setup_bullet_client(cfg.render)
    robot = instantiate(cfg.robot, bullet_client=bullet_client)
    t_bounds = copy.deepcopy(robot.workspace_bounds)
    t_bounds[2, 1] = t_bounds[2, 0]
    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    t_center = np.mean(t_bounds, axis=1)
    teletentric_camera = instantiate(cfg.teletentric_camera, bullet_client=bullet_client, t_center=t_center, robot=robot)

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

    state = env.reset()
    logger.debug(f"Environment reset.")

    logger.debug(f"Worker started!")
    while True:
        if not model_weights_queue.empty():
            model_weights = model_weights_queue.get()
            agent.model.set_weights(model_weights)

        action = agent.get_action(state)
        logger.debug(f"Action: {action}")
        next_state, reward, done, _ = env.step(action)
        replay_queue.put((state, action, reward, next_state, done))

        if done:
            state = env.reset()
            logger.debug("#################################################################################################")
            logger.info("Episode done, resetting environment")
            logger.debug("#################################################################################################")
        else:
            state = next_state

# Train the model using multiple environments
def train_model(env_config, num_envs):
    model_weights_queue = mp.Queue(maxsize=10)
    replay_queue = mp.Queue(maxsize=10000)
    replay_buffer = PrioritizedReplayBuffer()

    agent = DQNAgent(action_dim=2, input_shape=(84, 84, 4), weights_path=env_config.weights_path, weights_dir=env_config.weights_dir)
    logger.debug(f"Agent loaded.")

    processes = []
    for _ in range(num_envs):
        p = mp.Process(target=worker, args=(env_config, model_weights_queue, replay_queue, agent))
        p.start()
        processes.append(p)
    logger.debug(f"Processes started.")

    for episode in range(env_config.num_episodes):
        while not replay_queue.empty():
            replay_buffer.put(*replay_queue.get())

        if replay_buffer.size() >= env_config.batch_size:
            loss = agent.train(replay_buffer, env_config.batch_size)

        if episode % env_config.target_update_freq == 0:
            agent.update_target()

        if episode % env_config.save_freq == 0:
            agent.model.save_weights(f"{env_config.model_dir}/dqn_episode_{episode}", save_format="tf")

        model_weights_queue.put(agent.model.get_weights())

    for p in processes:
        p.terminate()
        p.join()

# Main function to start training
@hydra.main(version_base=None, config_path="config", config_name="DQN")
def main(cfg: DictConfig):
    train_model(cfg, num_envs=cfg.num_envs)

if __name__ == "__main__":
    main()
