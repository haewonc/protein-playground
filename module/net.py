import torch
from torch import nn, einsum
import torch.nn.functional as F

def elu():
    return nn.ELU(inplace=True)

def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)

def conv2d(in_chan, out_chan, kernel_size, dilation=1, **kwargs):
    padding = dilation * (kernel_size - 1) // 2
    return nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding, dilation=dilation, **kwargs)

class trRosetta(nn.Module):
    '''
    Reference: https://github.com/lucidrains/tr-rosetta-pytorch
    '''
    def __init__(self, filters=64, kernel=3, num_layers=61):
        super().__init__()
        self.filters = filters
        self.kernel = kernel
        self.num_layers = num_layers

        self.first_block = nn.Sequential(
            conv2d(442 + 2 * 42, filters, 1),
            instance_norm(filters),
            elu()
        )

        # stack of residual blocks with dilations
        cycle_dilations = [1, 2, 4, 8, 16]
        dilations = [cycle_dilations[i % len(cycle_dilations)] for i in range(num_layers)]

        self.layers = nn.ModuleList([nn.Sequential(
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters),
            elu(),
            nn.Dropout(p=0.15),
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters)
        ) for dilation in dilations])

        self.activate = elu()

        # conv to anglegrams and distograms
        self.to_prob_theta = nn.Sequential(conv2d(filters, 25, 1), nn.Softmax(dim=1))
        self.to_prob_phi = nn.Sequential(conv2d(filters, 13, 1), nn.Softmax(dim=1))
        self.to_distance = nn.Sequential(conv2d(filters, 37, 1), nn.Softmax(dim=1))
        self.to_prob_bb = nn.Sequential(conv2d(filters, 3, 1), nn.Softmax(dim=1))
        self.to_prob_omega = nn.Sequential(conv2d(filters, 25, 1), nn.Softmax(dim=1))
 
    def forward(self, x):
        x = self.first_block(x)

        for layer in self.layers:
            x = self.activate(x + layer(x))
        
        prob_theta = self.to_prob_theta(x)      # anglegrams for theta
        prob_phi = self.to_prob_phi(x)          # anglegrams for phi

        x = 0.5 * (x + x.permute((0,1,3,2)))    # symmetrize

        prob_distance = self.to_distance(x)     # distograms
        # prob_bb = self.to_prob_bb(x)            # beta-strand pairings (not used)
        prob_omega = self.to_prob_omega(x)      # anglegrams for omega

        return prob_theta, prob_phi, prob_distance, prob_omega


class trBackground(nn.Module):
    def __init__(self):
        super().__init__()
        n2d_layers = 36
        n2d_filters = 64
        window2d = 3

        self.first_block = nn.Sequential(
            conv2d(442 + 2 * 42, n2d_filters, 1),
            instance_norm(n2d_filters),
            elu()
        )

        dilation = 1
        layers = []
        for _ in range(n2d_layers):
            layers.append(conv2d(n2d_filters, n2d_filters, window2d, dilation=dilation))
            layers.append(instance_norm(n2d_filters))
            layers.append(elu())
            layers.append(nn.Dropout(p=0.15))
            layers.append(conv2d(n2d_filters, n2d_filters, window2d, dilation=dilation))
            layers.append(instance_norm(n2d_filters))
            layers.append(elu())
            dilation *= 2
            if dilation > 16:
                dilation = 1
        self.layers = nn.Sequential(*layers)

        self.to_prob_theta = nn.Sequential(conv2d(n2d_filters, 25, 1), nn.Softmax(dim=1))
        self.to_prob_phi = nn.Sequential(conv2d(n2d_filters, 13, 1), nn.Softmax(dim=1))
        self.to_distance = nn.Sequential(conv2d(n2d_filters, 37, 1), nn.Softmax(dim=1))
        self.to_prob_bb = nn.Sequential(conv2d(n2d_filters, 3, 1), nn.Softmax(dim=1))
        self.to_prob_omega = nn.Sequential(conv2d(n2d_filters, 25, 1), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.first_block(x)
        x = self.layers(x) # no skip connection 
        prob_theta = self.to_prob_theta(x)      # anglegrams for theta
        prob_phi = self.to_prob_phi(x)          # anglegrams for phi

        x = 0.5 * (x + x.permute((0,1,3,2)))    # symmetrize

        prob_distance = self.to_distance(x)     # distograms
        # prob_bb = self.to_prob_bb(x)          # beta-strand pairings (not used)
        prob_omega = self.to_prob_omega(x)      # anglegrams for omega

        return prob_theta, prob_phi, prob_distance, prob_omega