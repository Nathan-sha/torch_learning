
import torch.nn as nn

class MLPCustom(nn.modules):
    def __init__(self, input_dim: int = 2, output_dim: int = 2) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
        )


    def forward(self, x):
        return  self.net(x)