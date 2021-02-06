from typing import List
import torch
import torch.nn as nn
from scipy.stats import norm


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

        return loss


class ICP(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        preds = preds.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        return ((target > preds[:, :, 0]) & (target < preds[:, :, -1])).mean()


class MIL(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        return (preds[:, :, 0] - preds[:, :, -1]).abs().mean()
