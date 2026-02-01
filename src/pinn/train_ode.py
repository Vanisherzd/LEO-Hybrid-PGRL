import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from src.models.neural_force import OrbitalForceNet
from src.physics.ode_solver import integrate_trajectory

# --- Hardware Optimization ---
torch.set_float32_matmul_precision('high') 

class TrajectoryDataset(Dataset):
    def __init__(self, t, states, seq_len=60):
        # We need continuous sequences for ODE integration
        # t: (N,)
        # states: (N, 6)
        
        self.t = torch.tensor(t, dtype=torch.float32)
        self.states = torch.tensor(states, dtype=torch.float32)
        self.seq_len = seq_len
        
        # Valid start indices (must have seq_len steps ahead)
        self.valid_starts = range(0, len(t) - seq_len - 1, 10) # Stride 10 to reduce redundancy

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        end_idx = start_idx + self.seq_len + 1 # +1 to include t_0 to t_N
        
        # Segment
        traj_segment = self.states[start_idx : end_idx] # (Steps+1, 6)
        
        # Calculate mean dt for this segment (assumed roughly constant)
        t_segment = self.t[start_idx : end_idx]
        dt = (t_segment[-1] - t_segment[0]) / self.seq_len
        
        return traj_segment, dt

def train_ode():
    # Load Data
    data_path = os.path.join("data", "real_training_data.npz")
    data = np.load(data_path)
    t_norm = data['t']
    states_norm = data['states']
    
    # 80% Train
    split_idx = int(0.8 * len(t_norm))
    train_dataset = TrajectoryDataset(t_norm[:split_idx], states_norm[:split_idx], seq_len=60) # 60 steps integration window
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Neural ODE Training on {device}")
    
    # Model: Force Field
    model = OrbitalForceNet(input_dim=6, output_dim=3, hidden_dim=256).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 100
    
    print(f"Starting Training (SeqLen=60, Epochs={epochs})...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for traj_batch, dt_batch in train_loader:
            # traj_batch: (B, Steps+1, 6)
            # dt_batch: (B,) -- we use mean dt (scalar if uniform)
            
            traj_batch = traj_batch.to(device)
            # Initial state: (B, 6)
            initial_state = traj_batch[:, 0, :]
            
            # Use scalar dt for integration (assuming constant step size in training data)
            # Or use batch of dt? rk4_step supports dt as scalar.
            # Let's take mean of batch dt
            dt_step = dt_batch.mean().item()
            
            # Predict
            with torch.amp.autocast('cuda'):
                # Integrate forward S=60 steps
                # Output: (B, S+1, 6)
                pred_traj = integrate_trajectory(initial_state, dt_step, model, steps=60)
                
                # Loss over entire trajectory
                loss = nn.MSELoss()(pred_traj, traj_batch)
                
                # Regularize force magnitude? 
                # acc = model(traj_batch.view(-1, 6))
                # reg = 1e-6 * acc.pow(2).mean()
                # loss += reg
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Trajectory Loss: {total_loss/len(train_loader):.8f}")
            
    torch.save(model.state_dict(), os.path.join("weights", "f5_neural_ode_v6.pth"))
    print("Neural ODE model saved to f5_neural_ode_v6.pth")

if __name__ == "__main__":
    train_ode()
