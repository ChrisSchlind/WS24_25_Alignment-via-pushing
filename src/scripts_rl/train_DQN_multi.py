import sys
import copy
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import tensorflow as tf
import random
from collections import deque
import cv2
import multiprocessing as mp

from bullet_env.util import setup_bullet_client, stdout_redirected
from transform.affine import Affine
from bullet_env.ur10_cell import UR10Cell
from bullet_env.pushing_env import PushingEnv

# Define the Convolutional Deep Q-Network model
class ConvDQN(tf.keras.Model):
    def __init__(self, action_dim=2):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, 3, strides=2, activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(action_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return tf.nn.tanh(x)

# Define the Replay Buffer for storing experience tuples
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

# Define the DQN Agent
class DQNAgent:
    def __init__(self, action_dim, epsilon=0.1, gamma=0.99, input_shape=(84, 84, 4)):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_shape = input_shape

        # Initialize the main and target networks
        self.model = ConvDQN(action_dim)
        dummy_state = np.zeros((1,) + input_shape)
        self.model(dummy_state)

        self.target_model = ConvDQN(action_dim)
        self.target_model(dummy_state)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)

    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.uniform(-1, 1, self.action_dim)

        state = np.expand_dims(state, axis=0)
        action = self.model(state)[0].numpy()
        return np.clip(action, -1, 1)

    def train(self, replay_buffer, batch_size=32):
        if replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        targets = self.target_model(states).numpy()
        next_value = np.max(self.target_model(next_states).numpy(), axis=1)
        action_indices = np.argmax(actions, axis=1)
        targets[range(actions.shape[0]), action_indices] = rewards + (1 - dones) * next_value * self.gamma

        with tf.GradientTape() as tape:
            values = self.model(states)
            loss = tf.keras.metrics.mean_squared_error(targets, values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss.numpy()

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

# Function to run the environment
def run_environment(cfg, replay_buffer, agent, env_id):
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
    )

    for episode in range(cfg.num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(cfg.max_steps_per_episode):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.put(state, action, reward, next_state, done)

            if replay_buffer.size() >= cfg.batch_size:
                agent.train(replay_buffer, cfg.batch_size)

            if step % cfg.target_update_freq == 0:
                agent.update_target()

            state = next_state
            episode_reward += reward

            if done:
                break

        if cfg.debug:
            logger.info(f"Env {env_id} - Episode {episode}: Reward = {episode_reward}")

        if episode % cfg.save_freq == 0:
            agent.model.save_weights(f"{cfg.model_dir}/dqn_episode_{episode}")

    env.close()

# Main function to start the training
@hydra.main(version_base=None, config_path="config", config_name="DQN")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    action_dim = 2
    input_shape = (84, 84, 4)
    agent = DQNAgent(action_dim, input_shape=input_shape)
    replay_buffer = ReplayBuffer()

    num_envs = cfg.num_envs
    processes = []

    for env_id in range(num_envs):
        p = mp.Process(target=run_environment, args=(cfg, replay_buffer, agent, env_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if cfg.debug:
        logger.info("Training completed.")

if __name__ == "__main__":
    main()
