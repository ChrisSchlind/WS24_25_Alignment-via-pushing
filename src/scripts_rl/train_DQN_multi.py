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
    def __init__(self, action_dim, epsilon=0.1, gamma=0.99, input_shape=(84, 84, 4)):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_shape = input_shape

        self.model = ConvDQN(action_dim)
        dummy_state = np.zeros((1,) + input_shape)
        self.model(dummy_state)

        self.target_model = ConvDQN(action_dim)
        self.target_model(dummy_state)
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)

    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:  # epsilon-greedy policy makes the agent explore the environment
            random_action = np.random.uniform(-1, 1, self.action_dim)
            logger.debug("!!Random action: {}", random_action)
            return random_action

        # State is 84x84x4, RGB D   with 255,255,255,1 so we need to normalize it
        state[0] = state[0] / 255.0
        state[1] = state[1] / 255.0
        state[2] = state[2] / 255.0

        state = np.expand_dims(state, axis=0)

        action = self.model(state)[0].numpy()
        return action

    def train(self, replay_buffer, batch_size=32):
        if replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        targets = self.target_model(states).numpy()
        next_value = np.max(self.target_model(next_states).numpy(), axis=1)
        targets[np.arange(actions.shape[0]), np.argmax(actions, axis=1)] = rewards + (1 - dones) * next_value * self.gamma

        with tf.GradientTape() as tape:
            values = self.model(states)
            loss = tf.keras.losses.MSE(targets, values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss.numpy()

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
    replay_buffer = ReplayBuffer()

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
