"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
"""

import torch
import torch.nn as nn


class MAELoss(nn.Module):

    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, x, y):
        loss = torch.abs(x - y).mean(dim=1).mean(axis=0)
        return loss
