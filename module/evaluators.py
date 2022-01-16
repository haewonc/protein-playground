import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tarfile
from net import trRosetta
from abstract import Evaluator

class PQ_KL(Evaluator):
    def __init__(self,saved_dir="../saved_model/model.xaa.pt", device="cuda"):
        self.p = trRosetta().to(device)
        self.q = trRosetta().to(device)
        self.load_background(saved_dir)

    def load_background(self, saved_dir):
        self.q.load_state_dict(torch.load(saved_dir))