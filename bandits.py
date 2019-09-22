import numpy as np


class Bandit:
    """
    Multi-armed bandits
    """

    def __init__(self, k, data, batch_size=1000, p=None):
        self.k = k  # number of arms
        self.action_values = np.zeros(self.k)  # p for each arm
        self.optimal = 0  # best arm
        self.batch_size = batch_size
        # number of samples taken to compute p
        self.idx = np.zeros(self.k)
        # pointer for seeking the next batch start in each arm
        self.p = p  # p for each arm, starting point of action_values
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
        #import pdb; pdb.set_trace()
        start_idx = self.idx[action]
        end_idx = start_idx + self.batch_size + 1

        if end_idx <= df.shape[0]:
            rewards = self.compute_rewards(df.loc[start_idx:end_idx, :])
            self.idx[action] = end_idx
        else:
            rewards = 0
            self.done = True

        return rewards

    def compute_rewards(self, data):
        return data['success'].mean()
