import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(), # Tanh often better for physics/regression than ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return x + self.block(x)

class HybridPINN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, lstm_layers=1, resnet_blocks=3):
        """
        Args:
            input_dim: Dimension of state vector (x, y, z, vx, vy, vz) -> 6
            hidden_dim: Internal feature dimension
            lstm_layers: Number of LSTM layers
            resnet_blocks: Number of Residual Blocks in ResNet branch
        """
        super(HybridPINN, self).__init__()
        
        # --- Branch 1: LSTM (Temporal / Data) ---
        # Takes sequence of states: (Batch, Seq, Input_Dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        # We'll take the final hidden state as the embedding
        
        # --- Branch 2: ResNet (Spatial / Physics / Temporal Coordinate) ---
        # Takes time 't' (and potentially current state if we want)
        # To enable Autograd w.r.t 't' for physics, 't' must flow through here.
        # Let's map t (scalar) -> hidden_dim
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh()
        )
        
        self.resnet = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(resnet_blocks)]
        )
        
        # --- Fusion ---
        # Concatenate LSTM output (hidden_dim) + ResNet output (hidden_dim) -> 2*hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim) # Output: Predicted State (x, y, z, vx, vy, vz)
        )
        
        # --- Inverse Problem Parameter (Phase 5) ---
        # Learnable Log Ballistic Coefficient for Drag
        # Initial guess around -10 or 0? It's dimensionless. 
        # Drag is small, so coeff might be small. 
        # Let's start at 0.0, learning will adjust.
        self.log_ballistic_coeff = nn.Parameter(torch.tensor(0.0))

    def forward(self, t, history):
        """
        Args:
            t: Tensor of shape (Batch, 1) representing query time (normalized). 
               Must have requires_grad=True for Physics Loss.
            history: Tensor of shape (Batch, Seq_Len, Input_Dim) representing past states.
        """
        
        # Branch 1: LSTM
        # history input. LSTM output is (out, (h_n, c_n))
        # We use the final hidden state of the last layer: h_n[-1]
        # h_n shape: (num_layers, batch, hidden_dim)
        _, (h_n, _) = self.lstm(history)
        lstm_out = h_n[-1] # (Batch, Hidden_Dim)
        
        # Branch 2: ResNet (Physics/Time component)
        t_embed = self.time_encoder(t)
        resnet_out = self.resnet(t_embed) # (Batch, Hidden_Dim)
        
        # Fusion
        combined = torch.cat([lstm_out, resnet_out], dim=1)
        output = self.fusion(combined)
        
        return output

    def freeze_backbone(self):
        """
        Freezes the LSTM backbone for Transfer Learning (Phase 2).
        """
        print("Freezing LSTM backbone...")
        for param in self.lstm.parameters():
            param.requires_grad = False
            
if __name__ == "__main__":
    # Sanity Check
    model = HybridPINN()
    print(model)
    
    # Fake data
    B, Seq, Dim = 32, 10, 6
    t_batch = torch.rand((B, 1), requires_grad=True)
    hist_batch = torch.randn((B, Seq, Dim))
    
    out = model(t_batch, hist_batch)
    print(f"Output shape: {out.shape}") # Should be (32, 6)
    
    # Check Autograd
    # Compute derivative of output w.r.t t (acceleration check)
    # We sum output to get a scalar for backward(), just testing graph connection
    loss = out.sum()
    loss.backward()
    print(f"t.grad present? {t_batch.grad is not None}")
    
    model.freeze_backbone()
    print("LSTM frozen.")
