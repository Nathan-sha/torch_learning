from turtle import forward
import torch.nn as nn
from torch.nn.modules import MaxPool1d, pooling

class CNNModel(nn.modules):
    def __init__(self, input_dim: int=2, output_dim: int=2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim,output_dim,5),
            nn.MaxPool2d(output_dim),
        )

    def forward(self, x):
        return self.net(x)