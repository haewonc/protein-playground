import torch 
import numpy as np
from utils import reweight, msa2pssm
from net import trRosetta, trBackground, Ensemble
from abstract import Evaluator
import tensorflow as tf

class PQ_KL(Evaluator):
    def __init__(self, pred_dirs, bkgrd_dirs, aa_weight, device="cuda"):
        self.p = self.load_state(trRosetta, pred_dirs)
        self.q = self.load_state(trBackground, bkgrd_dirs)
        self.aa_bkgr = np.array([0.07892653, 0.04979037, 0.0451488 , 0.0603382 , 0.01261332,
                                    0.03783883, 0.06592534, 0.07122109, 0.02324815, 0.05647807,
                                    0.09311339, 0.05980368, 0.02072943, 0.04145316, 0.04631926,
                                    0.06123779, 0.0547427 , 0.01489194, 0.03705282, 0.0691271 ])
        self.aa_weight = aa_weight
        self.device = device

    def load_state(self, net, saved_dir):
        if isinstance(saved_dir, list):
            model = Ensemble(net, saved_dir, self.device)
        else:
            model = net().to(self.device)
            model.load_state_dict(torch.load(saved_dir))
        return model
    
    def evaluate(self, x):
        ncol = x.size(1)
        preds = self.p.forward(x)
        bkgrds = self.q.forward(x)
        loss = []
        for pred, bkgrd in zip(preds, bkgrds):
            loss.append(torch.mean(torch.sum(pred*torch.log(pred/bkgrd), dim=-1)))
        aa_samp = torch.sum(x[0,:,:20], axis=0)/ncol + 1e-7
        aa_samp = aa_samp/torch.sum(aa_samp)
        loss_aa = torch.sum(aa_samp*torch.log(aa_samp/self.aa_bkgr))

        return sum(loss) + self.aa_weight*loss_aa