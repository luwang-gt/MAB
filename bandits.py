import numpy as np


class Bandit:
    """
    Multi-armed bandits class
    """

    def __init__(self, k, data, p=None):
        self.k = k  # number of arms
        self.action_values = np.zeros(self.k)  # p for each arm
        self.optimal = np.random.choice(range(self.k))  # best arm
        self.idx = np.zeros(self.k)
        # pointer for seeking the next batch start in each arm
        self.p = p  # prior p for each arm, starting point of action_values
        self.data = data
        self.done = False

    def reset(self):
        self.idx = np.zeros(self.k)
        if not self.p:
            self.action_values = np.random.uniform(self.k)
        else:
            self.action_values = self.p

        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        df = self.data[action]
        start_idx = self.idx[action]

        if start_idx < df.shape[0]:
            rewards = self.compute_rewards(df.loc[start_idx:start_idx, :])
            self.idx[action] = start_idx + 1
        else:
            rewards = 0
            self.done = True

        return rewards

    def compute_rewards(self, data):
        rewards = data.success.iloc[0]
        return rewards


class ContextualBandits:
    """
    Contextual bandits class
    """

    def __init__(self, num_actions, num_states, data, state_col, p=None):
        self.state = 0  # current state
        self.num_actions = num_actions  # number of actions, i.e. arms
        self.num_states = num_states  # number of states
        self.data = data
        self.state_col = state_col
        self.idx = np.zeros([self.num_states, self.num_actions])
        self.done = False
        self.p = p  # prior probability for states

    def getBandit(self):
        if not self.p:
            self.state = np.random.randint(0, self.num_states)
        else:
            self.state = np.random.choice(range(self.num_states), p=self.p)
        return self.state

    def pull(self, state, action):
        df = self.data[state][action]
        start_idx = self.idx[state][action]

        if start_idx < df.shape[0]:
            rewards = self.compute_rewards(df.loc[start_idx:start_idx, :])
            self.idx[state][action] = start_idx + 1
        else:
            rewards = 0
            self.done = True

        return rewards

    def compute_rewards(self, data):
        rewards = data.success.iloc[0]
        return rewards
