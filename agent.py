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


class ContextualAgent:
    def __init__(self, bandit, policy):
        self.policy = policy
        self.num_actions = bandit.num_actions
        self.num_states = bandit.num_states
        self.value_estimates = np.zeros([self.num_states, self.num_actions])
        self.action_attempts = np.zeros([self.num_states, self.num_actions])
        self.last_state = None
        self.last_action = None
        self.step = 0

    def reset(self):
        self.value_estimates = np.zeros([self.num_states, self.num_actions])
        self.action_attempts = np.zeros([self.num_states, self.num_actions])
        self.last_state = None
        self.last_action = None
        self.step = 0

    def choose(self, state):
        action = self.policy.choose(self, state)
        self.last_action = action
        self.last_state = state

        return action

    def observe(self, reward):
        self.action_attempts[self.last_state][self.last_action] += 1
        gamma = 1 / self.action_attempts[self.last_state][self.last_action]
        q = self.value_estimates[self.last_state][self.last_action]
        self.value_estimates[self.last_state][self.last_action]\
        += gamma * (reward - q)

        self.step += 1
