import torch
from torch import nn

import os

class PlayerNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.model = self.model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    def save(self, file_name='model.pth'):
        folder = './model'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_name = os.path.join(folder, file_name)
        torch.save(self.state_dict(), file_name)
