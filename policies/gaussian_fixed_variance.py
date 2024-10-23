import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianFixedVariancePolicy(nn.Module):
    def __init__(self, threshold: float = 0.001):
        super().__init__()
        self.threshold = threshold
        self.register_parameter(
            "mu", nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        )
        self.register_buffer("sigma", torch.tensor(0.0001, dtype=torch.float32))

    def forward(self):
        threshold_dist = Normal(self.mu, self.sigma)
        return threshold_dist
