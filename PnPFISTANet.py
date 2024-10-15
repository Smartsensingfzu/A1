# -*- coding: utf-8 -*-
"""
Created on June 17, 2020

ISTANet(shared network with 4 conv + ReLU) + regularized hyperparameters softplus(w*x + b). 
The intention is to make gradient step \mu and thresholding value \theta positive and monotonically decrease.

@author: XIANG
"""

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import os

# Initialize network weights
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

# External denoiser module (e.g., BM3D, DnCNN, etc.)
class ExternalDenoiser(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_res_blocks=10):
        super(ExternalDenoiser, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_res_blocks)])
        self.conv_out = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.conv_in(x)
        x = self.res_blocks(x)
        x = self.conv_out(x)
        x = x + self.skip_connection(identity)
        return x

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x + identity
        return x

# Basic block of FISTA-Net
class BasicBlock(nn.Module):
    def __init__(self, features=32):
        super(BasicBlock, self).__init__()
        self.Sp = nn.Softplus()

        self.conv_D = nn.Conv2d(1, features, (3, 3), stride=1, padding=1)
        self.conv1_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)

        self.conv1_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv_G = nn.Conv2d(features, 1, (3, 3), stride=1, padding=1)

    def forward(self, x, PhiTPhi, PhiTb, LTL, mask, lambda_step, soft_thr, denoiser):
        pnum = x.size()[2]
        x = x.view(x.size()[0], x.size()[1], pnum * pnum, -1)  # (batch_size, channel, pnum*pnum, 1)
        x = torch.squeeze(x, 1)
        x = torch.squeeze(x, 2).t()
        x = mask.mm(x)

        # Quadratic TV gradient descent
        x = x - self.Sp(lambda_step) * torch.inverse(PhiTPhi + 0.001 * LTL).mm(PhiTPhi.mm(x) - PhiTb - 0.001 * LTL.mm(x))

        x = torch.mm(mask.t(), x)
        x = x.view(pnum, pnum, -1)
        x = x.unsqueeze(0)
        x_input = x.permute(3, 0, 1, 2)

        x_D = self.conv_D(x_input)

        x = self.conv1_forward(x_D)
        x = F.relu(x)
        x = self.conv2_forward(x)
        x = F.relu(x)
        x = self.conv3_forward(x)
        x = F.relu(x)
        x_forward = self.conv4_forward(x)

        # Soft-thresholding block
        x_st = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.Sp(soft_thr)))

        x = self.conv1_backward(x_st)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_backward = self.conv4_backward(x)

        x_G = self.conv_G(x_backward)

        # Prediction output (skip connection); non-negative output
        x_pred = F.relu(x_input + x_G)

        # Apply the external denoiser
        x_denoised = denoiser(x_pred)

        # Compute symmetry loss
        x = self.conv1_backward(x_forward)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_D_est = self.conv4_backward(x)
        symloss = x_D_est - x_D

        return [x_denoised, symloss, x_st]

# PnP-FISTA-Net
class PnPFISTANet(nn.Module):
    def __init__(self, LayerNo, Phi, L, mask, denoiser):
        super(PnPFISTANet, self).__init__()
        self.LayerNo = LayerNo
        self.Phi = Phi
        self.L = L
        self.mask = mask
        self.denoiser = denoiser

        self.fcs = nn.ModuleList([BasicBlock(features=32) for _ in range(LayerNo)])
        initialize_weights(self)

        # Thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        # Gradient step
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        # Two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))

        self.Sp = nn.Softplus()

    def forward(self, x0, b):
        b = torch.squeeze(b, 1)
        b = torch.squeeze(b, 2)
        b = b.t()

        PhiTPhi = self.Phi.t().mm(self.Phi)
        PhiTb = self.Phi.t().mm(b)
        LTL = self.L.t().mm(self.L)

        # Initialize the result
        xold = x0
        y = xold
        layers_sym = []  # For computing symmetric loss
        layers_st = []   # For computing sparsity constraint
        xnews = []       # Iteration result
        xnews.append(xold)

        for i in range(self.LayerNo):
            theta_ = self.w_theta * i + self.b_theta
            mu_ = self.w_mu * i + self.b_mu
            xnew, layer_sym, layer_st = self.fcs[i](y, PhiTPhi, PhiTb, LTL, self.mask, mu_, theta_, self.denoiser)
            rho_ = (self.Sp(self.w_rho * i + self.b_rho) - self.Sp(self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)
            y = xnew + rho_ * (xnew - xold)  # Two-step update
            xold = xnew
            xnews.append(xnew)   # Iteration result
            layers_st.append(layer_st)
            layers_sym.append(layer_sym)

        return [xnew, layers_sym, layers_st]

