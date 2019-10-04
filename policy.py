import numpy as np


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy policy class
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))
        else:
            action = np.argmax(agent.value_estimates)
            # Randomly select when there are ties for best choice
            check = np.where(agent.value_estimates == action)[0]
            if len(check) == 0:
                return action
            else:
                return np.random.choice(check)

        return action


class ContextualEpsilonGreedyPolicy:
    """
    Epsilon-greedy policy for contextual bandits
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose(self, agent, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(agent.value_estimates.shape[-1])
        else:
            action = np.argmax(agent.value_estimates[state])
            check = np.where(agent.value_estimates[state] == action)[0]
            if len(check) == 0:
                return action
            else:
                return np.random.choice(check)

        return action
