from msilib import sequence
from sched import scheduler
import numpy as np
from module.mutators import RandomMutator
from abstract import Agent

class HallAgent(Agent):
    def __init__(self, sequence, aa=None):
        '''
        Method
            Initialize the hallucination agent
        Params
            sequence (numpy array) current amino acid sequence
            aa (numpy array) amino acid list used in sampling
        '''
        self.length = len(sequence)
        self.sequence = sequence
        self.mutator = RandomMutator(self.length, aa)
        if aa is not None:
            self.aa = aa 
        else:
            self.aa = np.arange(20)
    
    def mutate(self):
        '''
        Method
            Mutate the sequence
        '''
        pos, aa = self.mutator.mutate(self.sequence)
        self.sequence[pos] = aa
        return self.sequence

class trAgent(Agent):
    def __init__(self, sequence, aa=None, beta=10.0, multiplier=2, schedule=5000):
        '''
        Method
            Initialize the hallucination agent
        Params
            sequence (numpy array) current amino acid sequence
            aa (numpy array) amino acid list used in sampling
        '''
        self.length = len(sequence)
        self.mutator = RandomMutator(self.length, aa)
        self.beta = beta
        self.schedule = schedule
        self.multiplier = multiplier
        self.steps = 0
        self.E = 999.9
        self.sequence = self.random_seq(self.length)
        self.temporary = None
        if aa is not None:
            self.aa = aa 
        else:
            self.aa = np.arange(20)
    
    def mutate(self):
        '''
        Method
            Mutate the sequence
        '''
        pos, aa = self.mutator.mutate(self.sequence)
        self.temporary = np.copy(self.sequence)
        self.sequence[pos] = aa
        return self.sequence
    
    def report(self, E):
        '''
        Method
            Accept or discard the mutation
        Params
            E (float) result of mutation
        '''
        if E < self.E:
            self.E = E
        else:
            if np.exp((E-self.E)*self.beta) > np.random.uniform():
                self.E = E
            else:
                self.sequence = np.copy(self.temporary) 
                self.temporary = None     

        self.steps += 1
        if self.steps % self.schedule == self.schedule-1:
            self.beta *= self.multiplier
        
        return self.sequence