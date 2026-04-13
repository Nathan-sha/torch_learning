import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)
    
    



