import numpy as np
import gym

legend = {
    'empty': 0,
    'agent': 4,
    'blocking': 8
}


class GridWorld:
    def __init__(self, world_shape, agent_init_pos, blocking_states, terminal_states, reward_states):
        self.world_shape = world_shape
        self.agent_init_pos = agent_init_pos
        self.blocking_states = blocking_states
        self.terminal_states = terminal_states
        self.reward_states = reward_states

        # the action representations are now integers, to make indexing and sampling for TD learning simpler
        self.action_space = gym.spaces.Discrete(4)
        self.possible_actions = {
            0: np.array([-1, 0]),  # up
            1: np.array([1, 0]),  # down
            2: np.array([0, 1]),  # right
            3: np.array([0, -1])  # left
        }

        # set initial agent position
        self.agent_current_pos = self.agent_init_pos
        # list of collected rewards, to not collect rewards twice
        self.collected_rewards = []

    def reset(self):
        # reset agent position
        self.agent_current_pos = self.agent_init_pos
        # reset list of collected rewards
        self.collected_rewards = []

        # render initial observation
        observation = np.copy(self.agent_current_pos)
        return observation

    def move_agent(self, action):
        # move agent
        new_agent_pos = np.array(self.agent_current_pos) + self.possible_actions[action]

        # check if new position is blocked
        if tuple(new_agent_pos) in self.blocking_states:
            return self.agent_current_pos

        # check if new position is out of bounds
        if (new_agent_pos < 0).any() or (new_agent_pos >= self.world_shape).any():
            return self.agent_current_pos

        return tuple(new_agent_pos)

    def step(self, action):
        # execute action
        self.agent_current_pos = self.move_agent(action)

        reward = 0.0
        done = False

        # check if there is any reward
        if tuple(self.agent_current_pos) in self.reward_states.keys() and tuple(
                self.agent_current_pos) not in self.collected_rewards:
            reward += self.reward_states[tuple(self.agent_current_pos)]
            self.collected_rewards.append(tuple(self.agent_current_pos))

        # check if there is any reward and whether the game ended
        if tuple(self.agent_current_pos) in self.terminal_states:
            done = True

        # render observation
        observation = np.copy(self.agent_current_pos)
        return observation, reward, done

    def render(self, show_render=True):
        # initialize empty states
        states = np.ones(self.world_shape) * legend['empty']

        # add agent
        states[tuple(self.agent_current_pos)] = legend['agent']

        # add blocking states
        for blocking_state in self.blocking_states:
            states[blocking_state] = legend['blocking']

        # add rewards
        for state, reward in self.reward_states.items():
            if state not in self.collected_rewards:
                states[state] = reward
        if show_render:
            print(states)
        return states
