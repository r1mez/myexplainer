import torch
from torch import nn


class StructureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StructureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)