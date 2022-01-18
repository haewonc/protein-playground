from msilib import sequence
import numpy as np
from module.mutators import RandomMutator
from abstract import Agent

class HallAgent(Agent):
    def __init__(self, target, sequence=None, aa=None):
        '''
        Method
            Initialize the hallucination agent
        Params
            target (numpy array) target amino acid sequence
            sequence (numpy array) current amino acid sequence
            aa (numpy array) amino acid list used in sampling
        '''
        self.length = len(target)
        self.target = target
        self.mutator = RandomMutator(self.length, aa)
        if sequence is not None:
            self.sequence = sequence
        else:
            self.sequence = self.random_seq(self.length)
        if aa is not None:
            self.aa = aa 
        else:
            self.aa = np.array(range(21))
        
    def random_seq(self, length, aa=None):
        '''
        Method
            Randomly select amino acid sequence from amino acid pool 
        Params
            length (int) legnth of sequence
            aa (numpy array) amino acid list used in sampling
        '''
        return np.random.choice(self.aa if aa is None else aa, length)
    
    def mutate(self):
        '''
        Method
            Mutate the sequence
        '''
        pos, aa = self.mutator.mutate(self.sequence)
        self.sequence[pos] = aa
        return self.sequence

class trAgent(Agent):
    def __init__(self, target, sequence=None, aa=None, beta=10.0):
        '''
        Method
            Initialize the hallucination agent
        Params
            target (numpy array) target amino acid sequence
            sequence (numpy array) current amino acid sequence
            aa (numpy array) amino acid list used in sampling
        '''
        self.length = len(target)
        self.target = target
        self.mutator = RandomMutator(self.length, aa)
        self.beta = beta
        self.E = 999.9
        if sequence is not None:
            self.sequence = sequence
        else:
            self.sequence = self.random_seq(self.length)
        self.temporary = None
        if aa is not None:
            self.aa = aa 
        else:
            self.aa = np.array(range(21))
        
    def random_seq(self, length, aa=None):
        '''
        Method
            Randomly select amino acid sequence from amino acid pool 
        Params
            length (int) legnth of sequence
            aa (numpy array) amino acid list used in sampling
        '''
        return np.random.choice(self.aa if aa is None else aa, length)
    
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