import numpy as np
import random
from abstract import Mutator

class RandomMutator():
    def __init__(self, length, aa=None):
        self.length = length 
        self.population = np.arange(21)
        if aa is None:
            policy = np.ones(21)
        else:
            policy = np.zeros(21)
            for i in aa:
                self.policy[i] = 1.0
        self.policy = policy / np.sum(policy)

    def mutate(self, cur):
        policy = np.copy(self.policy)
        pos = random.randint(0, self.length-1)
        policy[cur[pos]] = 0.0
        return pos, random.choices(population=self.population, weights=policy, k=1)[0]