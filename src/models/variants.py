import torch
import torch.nn as nn
import math

class FeatureTokenizer(nn.Module):
    """
    Splits state (Batch, 6) into Position and Velocity tokens.
    Projects them to a latent space.
    Output: (Batch, Sequence=2, Hidden)
    """
    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()
        # We assume input is (rx, ry, rz, vx, vy, vz)
        self.r_proj = nn.Linear(3, hidden_dim)
        self.v_proj = nn.Linear(3, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, state):
        r = state[:, 0:3]
        v = state[:, 3:6]
        
        # Project
        r_emb = self.r_proj(r) # (Batch, Hidden)
        v_emb = self.v_proj(v) # (Batch, Hidden)
        
        # Stack as sequence [Position, Velocity]
        # (Batch, 2, Hidden)
        seq = torch.stack([r_emb, v_emb], dim=1)
        return seq

class OrbitalLSTM(nn.Module):
    def __init__(self, input_dim=6, output_dim=3, hidden_dim=128, num_layers=1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Init final layer
        nn.init.uniform_(self.head[-1].weight, -1e-5, 1e-5)
        nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, state):
        # (Batch, 2, Hidden)
        seq = self.tokenizer(state)
        
        # LSTM Output
        # output: (Batch, Seq, Hidden)
        # hn: (Layers, Batch, Hidden)
        output, (hn, cn) = self.lstm(seq)
        
        # We take the final hidden state (after processing Position THEN Velocity)
        # This implies Velocity "updates" the context established by Position.
        last_hidden = output[:, -1, :] # (Batch, Hidden)
        
        return self.head(last_hidden)

class OrbitalGRU(nn.Module):
    def __init__(self, input_dim=6, output_dim=3, hidden_dim=128, num_layers=1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, hidden_dim)
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Init final layer
        nn.init.uniform_(self.head[-1].weight, -1e-5, 1e-5)
        nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, state):
        seq = self.tokenizer(state)
        output, hn = self.gru(seq)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)

class OrbitalAttention(nn.Module):
    def __init__(self, input_dim=6, output_dim=3, hidden_dim=128, num_heads=4):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, hidden_dim)
        
        # Learnable Positional Encodings for 2 positions
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Flatten: 2 * Hidden -> Output
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Init final layer
        nn.init.uniform_(self.head[-1].weight, -1e-5, 1e-5)
        nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, state):
        seq = self.tokenizer(state)
        
        # Add Positional Embedding
        x = seq + self.pos_embedding
        
        # Self-Attention
        # attn_output: (Batch, Seq, Hidden)
        attn_output, _ = self.attn(x, x, x)
        
        # Residual Connection + Norm?
        # For simplicity in this benchmark, we pass the attn output directly
        # But commonly we do x = x + attn_output
        x = x + attn_output
        
        # Flatten
        flat = x.reshape(x.size(0), -1) # (Batch, 2*Hidden)
        
        return self.head(flat)
