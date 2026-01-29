import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.models.neural_force import OrbitalForceNet
from src.utils import Normalizer
from src.physics.ode_solver import integrate_trajectory

# --- Configuration ---
DATA_PATH = os.path.join("data", "precise_training_data.npz")
SAVE_PATH = os.path.join("weights", "f5_sp3_precision.pth")
EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 1024 # Batching points if we use non-sequential training, but for ODE we usually do seq.
# For Neural ODE, we often train on short snippets.

def train_sp3():
    print("--- Training SP3 Precision Model ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # 1. Load Data
    data = np.load(DATA_PATH)
    t_norm = torch.tensor(data['t'], dtype=torch.float32).to(device)
    states_norm = torch.tensor(data['states'], dtype=torch.float32).to(device)
    
    print(f"Data Loaded: {len(t_norm)} points")
    
    # 2. Model Init (Fresh Brain)
    model = OrbitalForceNet(input_dim=6, output_dim=3, hidden_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 3. Training Loop (Simple Trajectory Fitting)
    # We will fit 60-step chunks for stability
    chunk_size = 60
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        # Random sampling of chunks
        # In a real rigorous training, we'd use a DataLoader
        # Here we just grab 10 random chunks per epoch for speed demonstration
        
        num_chunks = 10
        for _ in range(num_chunks):
            start_idx = np.random.randint(0, len(t_norm) - chunk_size - 1)
            end_idx = start_idx + chunk_size
            
            t_chunk = t_norm[start_idx:end_idx]
            state_chunk = states_norm[start_idx:end_idx]
            
            # Initial Condition
            x0 = state_chunk[0:1] # Shape (1, 6)
            
            # Integrate
            # Need dt. Assuming uniform steps in t_norm
            dt = t_chunk[1] - t_chunk[0]
            pred_states = integrate_trajectory(x0, dt, model, steps=chunk_size-1)
            
            # Loss (MSE)
            loss = nn.MSELoss()(pred_states, state_chunk.unsqueeze(0))
            loss.backward()
            total_loss += loss.item()
            
        optimizer.step()
        scheduler.step(total_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/num_chunks:.8f}")
            
    # 4. Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_sp3()
