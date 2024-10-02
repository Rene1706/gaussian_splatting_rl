import torch
import torch.nn as nn


class SigmoidNormThresholdPolicy(nn.Module):
    def forward(self, x, threshold):
        return torch.sigmoid(torch.norm(x, dim=-1, p=2) - threshold) > 0.5
