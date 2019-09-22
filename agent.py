import numpy as np


class Agent:
    def __init__(self, bandit, policy):
        self.policy = policy
        self.k = bandit.k
        self.value_estimates = np.zeros(self.k)
        self.action_attempts = np.zeros(self.k)
        self.last_action = None
        self.step = 0

    def reset(self):
        self.value_estimates = np.zeros(self.k)
        self.action_attempts = np.zeros(self.k)
        self.last_action = None
        self.step = 0

    def choose(self):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1
        gamma = 1 / self.action_attempts[self.last_action]
        q = self.value_estimates[self.last_action]
        self.value_estimates[self.last_action] += gamma * (reward - q)
        # running average of rewards
        self.step += 1

