import numpy as np
import random
from abstract import Mutator

class RandomMutator():
    def __init__(self, length, remove_aa=None):
        self.length = length 
        self.population = np.arange(20)
        policy = np.ones(20)
        if remove_aa is not None:
            for i in remove_aa:
                self.policy[i] = 0.0
        self.policy = policy / np.sum(policy)

    def mutate(self, cur):
        policy = self.policy + 0.0
        pos = random.randint(0, self.length-1)
        policy[cur[pos]] = 0.0
        return pos, random.choices(population=self.population, weights=policy, k=1)[0]