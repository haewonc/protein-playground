from abc import * 

class Agent(metaclass=ABCMeta):
    @abstractmethod
    def mutate(self):
        pass

class Mutator(metaclass=ABCMeta):
    @abstractmethod
    def initialize(self):
        pass
    def fit(self):
        pass
    def mutate(self):
        pass

class Evaluator(metaclass=ABCMeta):
    def evaluate(self):
        pass