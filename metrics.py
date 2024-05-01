"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
"""

import torch


def calculate_metrics(x, y):
    x = torch.real(x)
    y = torch.real(y)

    mse = torch.square(x - y).mean(dim=1).mean(dim=0)

    mae = torch.abs(x - y).mean(dim=1).mean(dim=0)

    output = {
        "mse": mse.item(),
        "mae": mae.item(),
    }

    return output
