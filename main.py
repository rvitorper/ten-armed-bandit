import numpy as np

sigma = 1  # for all slots
# 10 different means
slots = {
    1: -1.0,
    2: 3.0,
    3: 1.4,
    4: 2.7,
    5: 3.2,
    6: 0.3,
    7: 0.9,
    8: 1.0,
    9: 1.1,
    10: 0.0
}


def bandit(index):
    mean = slots[index]
    return np.random.normal(mean)


class Agent:
    def __init__(self):
        self.value_estimates = {}
        self.slot_sample_count = {}
        for i in range(1, 11):
            self.value_estimates[i] = 0.0
            self.slot_sample_count[i] = 0.0
        self.rewards = []
        self.slot_index = []
        self.slot_optimal = []

    def _sample(self, slot_index=-1):
        reward = bandit(slot_index)
        self.slot_index.append(slot_index)
        self.slot_optimal.append(1.0 if slot_index == 5 else 0.0)
        self.rewards.append(reward)
        self.slot_sample_count[slot_index] = self.slot_sample_count[slot_index] + 1.0
        self.value_estimates[slot_index] = self.value_estimates[slot_index] + \
                                           (reward - self.value_estimates[slot_index]) / self.slot_sample_count[slot_index]

    def __str__(self):
        return 'value: ' + str(sum(self.rewards) / len(self.rewards)) + ' probability of getting optimal case: ' + \
               str(float(sum(self.slot_optimal)) / len(self.slot_index))


class Greedy(Agent):
    def __init__(self):
        super().__init__()

    def sample(self):
        m = max(self.value_estimates.values())
        to_sample = [key for key, value in self.value_estimates.items() if value == m]
        i = np.random.randint(0, len(to_sample))
        slot_index = to_sample[i]
        super()._sample(slot_index)


class EpsilonGreedy(Agent):
    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def epsilon_case(self):
        i = np.random.randint(1, 11)
        super()._sample(slot_index=i)

    def greedy_case(self):
        m = max(self.value_estimates.values())
        to_sample = [key for key, value in self.value_estimates.items() if value == m]
        i = np.random.randint(0, len(to_sample))
        slot_index = to_sample[i]
        super()._sample(slot_index)

    def sample(self):
        sampled_number = np.random.sample()
        if sampled_number <= self.epsilon:
            self.epsilon_case()
        else:
            self.greedy_case()


epsilon_greedy_ten = EpsilonGreedy(epsilon=0.1)
epsilon_greedy_one = EpsilonGreedy(epsilon=0.01)
greedy = Greedy()

sample_size = 100000
for i in range(sample_size):
    epsilon_greedy_ten.sample()
    epsilon_greedy_one.sample()
    greedy.sample()

print(greedy)
print(epsilon_greedy_one)
print(epsilon_greedy_ten)
