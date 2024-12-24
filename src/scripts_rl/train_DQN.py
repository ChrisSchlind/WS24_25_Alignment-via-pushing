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

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*sample))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, action_dim, epsilon=0.1, input_shape=(84, 84, 4)):
        self.action_dim = action_dim
        self.epsilon = epsilon
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
            return np.random.uniform(-1, 1, self.action_dim)

        state = np.expand_dims(state, axis=0)
        # Direct continuous output from network
        action = self.model(state)[0].numpy()
        # Ensure actions are in [-1,1] range
        return np.clip(action, -1, 1)

    def train(self, replay_buffer, batch_size=32, gamma=0.99):
        if replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        next_q_values = self.target_model(next_states)
        target_q = rewards + (1 - dones) * gamma * tf.reduce_max(next_q_values, axis=1)

        with tf.GradientTape() as tape:
            predicted_actions = self.model(states)
            # MSE loss for continuous actions
            loss = tf.reduce_mean(tf.square(actions - predicted_actions))

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss.numpy()

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
    camera_factory = instantiate(cfg.camera_factory, bullet_client=bullet_client, t_center=t_center)
    teletentric_camera = instantiate(cfg.teletentric_camera, bullet_client=bullet_client, t_center=t_center)

    # Create environment with all components
    env = PushingEnv(
        bullet_client=bullet_client,
        robot=robot,
        task_factory=task_factory,
        teletentric_camera=teletentric_camera,
        workspace_bounds=cfg.workspace_bounds,
    )

    logger.info("Instantiation completed.")

    # Initialize DQN agent with 2D continuous action space
    action_dim = 2  # (x,y) continuous actions
    input_shape = (84, 84, 4)  # RGB (3) + depth (1) = 4 channels
    agent = DQNAgent(action_dim, input_shape=input_shape)
    logger.info("DQN agent initialized.")
    replay_buffer = ReplayBuffer()
    logger.info("Replay buffer initialized.")

    # Training loop
    for episode in range(cfg.num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(cfg.max_steps_per_episode):
            action = agent.get_action(state)

            # Get next state using environment's step function
            next_state, reward, done, _ = env.step(action)

            replay_buffer.put(state, action, reward, next_state, done)

            if replay_buffer.size() >= cfg.batch_size:
                loss = agent.train(replay_buffer, cfg.batch_size)

            if step % cfg.target_update_freq == 0:
                agent.update_target()

            state = next_state
            episode_reward += reward

            if done:
                break

        logger.info(f"Episode {episode}: Reward = {episode_reward}")

        # Save model periodically
        if episode % cfg.save_freq == 0:
            agent.model.save_weights(f"{cfg.model_dir}/dqn_episode_{episode}")

    env.close()


if __name__ == "__main__":
    main()
