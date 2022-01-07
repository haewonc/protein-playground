from module.abstract import Agent
import numpy as np

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
        if sequence is not None:
            self.sequence = sequence
        else:
            self.sequence = self.random_seq(self.length)
        if aa is not None:
            self.aa = aa 
        else:
            self.aa = np.array(range(20))
        
    def random_seq(self, length, aa=None):
        '''
        Method
            Randomly select amino acid sequence from amino acid pool 
        Params
            length (int) legnth of sequence
            aa (numpy array) amino acid list used in sampling
        '''
        return np.random.choice(self.aa if aa is None else aa, length)
    
    def mutate(self, pos, aa):
        '''
        Method
            Mutate the sequence
        Params
            pos (int) position of mutation
            aa (int) selected amino acid 
        '''
        self.sequence[pos] = aa