import numpy as np


class Policy:
    """
    General policy class
    """

    def __str__(self):
        return "Policy"


class EpsilonGreedyPolicy(Policy):
    """
    Epsilon-greedy policy class
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return "Epsilon-greedy policy"

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
