import torch
import torch.nn as nn

class OrbitalForceNet(nn.Module):
    def __init__(self, input_dim=6, output_dim=3, hidden_dim=256):
        super(OrbitalForceNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize final layer with small weights
        # This ensures the model starts by predicting ~0 perturbation
        # allowing the J2 analytical physics to dominate initially.
        nn.init.uniform_(self.net[-1].weight, -1e-5, 1e-5)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, state):
        """
        Args:
            state: (Batch, 6) [rx, ry, rz, vx, vy, vz] normalised.
        Returns:
            acc_perturbation: (Batch, 3) [ax, ay, az] normalised.
        """
        return self.net(state)
