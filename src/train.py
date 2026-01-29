import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from src.model import HybridPINN

# --- Hardware Optimization ---
torch.set_float32_matmul_precision('high') # For RTX 3000/4000/5000 (Ampere/Blackwell TF32)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from src.model import HybridPINN
from src.utils import J2 # Dimensionless J2

# --- Hardware Optimization ---
torch.set_float32_matmul_precision('high') 

def j2_acceleration_dimensionless(position):
    """
    Computes analytical gravitational acceleration in DIMENSIONLESS units.
    GM = 1.0 (normalized)
    R_EARTH = 1.0 (normalized)
    
    Args:
        position: Tensor (Batch, 3) [x, y, z] (Normalized)
    Returns:
        acceleration: Tensor (Batch, 3) [ax, ay, az] (Normalized)
    """
    x = position[:, 0:1]
    y = position[:, 1:2]
    z = position[:, 2:3]
    
    r = torch.sqrt(x**2 + y**2 + z**2)
    r2 = r**2
    
    # In dimensionless units:
    # a_monopole = - mu / r^3 * r  (mu=1) => - r / r^3
    # J2 perturbations:
    # a_J2_x = - (3/2) * J2 * (1/r^2) * (1/r)^2 * (x/r) * (5*(z/r)^2 - 1)
    #        = - (3/2) * J2 * (x / r^5) * (5*z^2/r^2 - 1)
    
    # Precompute common factors
    # r^5 for J2 term
    r5 = r**5
    r3 = r**3
    
    factor_J2 = (1.5 * J2) / r5
    z2_r2 = (z / r)**2
    
    # Monopole + J2
    # ax = -x/r^3 + factor_J2 * x * (5*z^2/r^2 - 1)
    # But wait, the standard form:
    # a = -mu/r^3 * r + ...
    # J2 term usually defined with (Re/r)^2. Since Re=1 in DU, it is (1/r)^2.
    
    ax = - (x / r3) + factor_J2 * x * (5 * z2_r2 - 1)
    ay = - (y / r3) + factor_J2 * y * (5 * z2_r2 - 1)
    az = - (z / r3) + factor_J2 * z * (5 * z2_r2 - 3)
    
    return torch.cat([ax, ay, az], dim=1)

# --- Dataset ---
class OrbitDataset(Dataset):
    def __init__(self, t, states, seq_len=10):
        self.t = torch.tensor(t, dtype=torch.float32).unsqueeze(1) # (N, 1)
        self.states = torch.tensor(states, dtype=torch.float32)    # (N, 6)
        self.seq_len = seq_len
        self.valid_indices = range(seq_len, len(t))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        # History window
        history = self.states[target_idx - self.seq_len : target_idx]
        target_state = self.states[target_idx]
        target_t = self.t[target_idx]
        return target_t, history, target_state

# --- Training Loop ---
def train_pinn():
    # Load Data (Normalized)
    data_path = os.path.join("data", "real_training_data.npz")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data_fetcher.py first.")
        return

    data = np.load(data_path)
    t_norm = data['t']
    states_norm = data['states']
    
    # Split Train/Val
    split_idx = int(0.8 * len(t_norm))
    train_dataset = OrbitDataset(t_norm[:split_idx], states_norm[:split_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True) # Increased batch size for 5080
    
    # Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} ({torch.cuda.get_device_name(0)})")
    
    model = HybridPINN(input_dim=6, hidden_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    epochs = 50 
    lambda_phys = 1e-4 # Physics loss can be small, weight it appropriately. 
                       # In dimensionless units, values are O(1).
    
    print("Starting Training (Dimensionless)...")
    scaler = torch.amp.GradScaler('cuda') 
    
    for epoch in range(epochs):
        model.train()
        total_data_loss = 0
        total_phys_loss = 0
        
        for t_batch, hist_batch, target_batch in train_loader:
            t_batch = t_batch.to(device).requires_grad_(True)
            hist_batch = hist_batch.to(device)
            target_batch = target_batch.to(device)
            
            with torch.amp.autocast('cuda'):
                pred_state = model(t_batch, hist_batch)
                
                # 1. Data Loss
                loss_data = nn.MSELoss()(pred_state, target_batch)
                
                # 2. Physics Loss (Dimensionless)
                x = pred_state[:, 0:1]
                y = pred_state[:, 1:2]
                z = pred_state[:, 2:3]
                
                ones = torch.ones_like(x)
                
                # Velocity (d r_tilde / d t_tilde)
                vx = torch.autograd.grad(x, t_batch, grad_outputs=ones, create_graph=True)[0]
                vy = torch.autograd.grad(y, t_batch, grad_outputs=ones, create_graph=True)[0]
                vz = torch.autograd.grad(z, t_batch, grad_outputs=ones, create_graph=True)[0]
                
                # Acceleration (d v_tilde / d t_tilde)
                ax = torch.autograd.grad(vx, t_batch, grad_outputs=ones, create_graph=True)[0]
                ay = torch.autograd.grad(vy, t_batch, grad_outputs=ones, create_graph=True)[0]
                az = torch.autograd.grad(vz, t_batch, grad_outputs=ones, create_graph=True)[0]
                
                pred_acc = torch.cat([ax, ay, az], dim=1)
                
                # Analytical J2 Gravity (Dimensionless)
                acc_physics = j2_acceleration_dimensionless(pred_state[:, 0:3])
                
                loss_physics = nn.MSELoss()(pred_acc, acc_physics)
                
                # Kinematic: v_net vs dx/dt
                pred_vel_net = pred_state[:, 3:6]
                calc_vel = torch.cat([vx, vy, vz], dim=1)
                loss_kinematic = nn.MSELoss()(pred_vel_net, calc_vel)
                
                loss_phys_total = loss_physics + loss_kinematic

                loss = loss_data + lambda_phys * loss_phys_total

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_data_loss += loss_data.item()
            total_phys_loss += loss_phys_total.item()
            
        avg_data_loss = total_data_loss / len(train_loader)
        avg_phys_loss = total_phys_loss / len(train_loader)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Data Loss: {avg_data_loss:.6f} | Physics Loss: {avg_phys_loss:.6f}")

    torch.save(model.state_dict(), os.path.join("weights", "f5_realdata_v1.pth"))
    print("Model saved to f5_realdata_v1.pth")

if __name__ == "__main__":
    train_pinn()
