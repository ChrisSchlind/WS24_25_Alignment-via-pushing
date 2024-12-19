import numpy as np
import tensorflow as tf
from collections import deque
import random


class DQN(tf.keras.layers.Layer):
    def __init__(self, units=(100, 50), n_actions=2, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.layers = []
        for u in units:
            self.layers.append(tf.keras.layers.Dense(u, activation='relu'))
        self.layers.append(tf.keras.layers.Dense(n_actions))

    def call(self, inputs, **kwargs):
        outputs = inputs
        for l in self.layers:
            outputs = l(outputs)
        return outputs


class DQNAgent:
    def __init__(self, action_space, observation_shape, epsilon=0.9, gamma=0.95):
        self.action_space = action_space
        # TODO this is only valid for discrete action spaces
        self.model = DQN(n_actions=self.action_space.n)
        self.target_model = DQN(n_actions=self.action_space.n)
        self.epsilon = epsilon  # exploration vs exploitation
        self.gamma = gamma  # discount factor
        self.optimizer = tf.keras.optimizers.Adam()
        self._init_networks(observation_shape)

    def _init_networks(self, observation_shape):
        initializer = np.zeros([1, *observation_shape])
        self.model(initializer)
        self.target_model(initializer)
        self.target_update()

    def act(self, observation, explore=True):
        if explore:
            if np.random.uniform(0, 1) < self.epsilon:
                action = self.action_space.sample()
                return action
        values = np.squeeze(self.model(observation).numpy())
        return np.argmax(values)

    def target_update(self):
        # TODO: set discount factor
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def learn(self, states, actions, rewards, next_states, dones):
        targets = self.target_model(states).numpy()
        next_value = self.target_model(next_states).numpy().max(axis=1)
        # new approximation of the action value; the '1-dones' means, that if the game
        # was terminated, there is no next state, thus also no future rewards
        targets[range(actions.shape[0]), actions] = rewards + (1 - dones) * next_value * self.gamma
        with tf.GradientTape() as tape:
            values = self.model(states)
            # MSE loss
            loss = tf.math.reduce_mean(tf.math.square(targets - values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss.numpy()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size=1):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*sample))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)
