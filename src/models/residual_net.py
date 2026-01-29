import torch
import torch.nn as nn

class SGP4ErrorCorrector(nn.Module):
    """
    MLP architecture to model the residual error between SGP4 and SP3 ground truth.
    Inputs: Normalized State (r, v) from SGP4.
    Outputs: Position Correction (delta_x, delta_y, delta_z).
    """
    def __init__(self, input_dim=6, output_dim=3, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, state):
        """
        state: (Batch, 6) -> [r_norm, v_norm]
        returns: (Batch, 3) -> [dr_x, dr_y, dr_z]
        """
        return self.net(state)
