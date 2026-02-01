import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from src.models.neural_force import OrbitalForceNet
from src.physics.golden_solver import GoldenDynamics, rk4_step_golden
from src.physics.advanced_forces import CelestialEphemeris

# --- Configuration ---
DATA_PATH = os.path.join("data", "precise_training_data.npz")
SAVE_PATH = os.path.join("weights", "f5_golden_v1.pth")
EPOCHS = 100
LR = 1e-4
CHUNK_SIZE = 60 # Steps per batch integration

def train_golden():
    print("--- Training Golden Physics Model (SP3) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print("Data not found. Please run ingest_sp3.py.")
        return
        
    data = np.load(DATA_PATH)
    # Use RAW states (km, km/s) because Golden Solver is Physical
    # But wait, t_raw is seconds?
    # t_norm and states_norm were used for v1.
    # For Golden Solver, we operate in PHYSICAL units for precision physics,
    # then maybe normalize for NN input inside wrapper?
    # Or just train NN to output physical residual (small number).
    
    # Let's trust the loaded "states_raw" if available, else denormalize.
    if 'states_raw' in data:
        t_sec = torch.tensor(data['t_raw'], dtype=torch.float32).to(device)
        states = torch.tensor(data['states_raw'], dtype=torch.float32).to(device)
        # Epoch
        start_epoch = float(data['start_epoch']) if 'start_epoch' in data else time.time() # Fallback
    else:
        print("Error: 'states_raw' (Physical units) required for Golden Physics.")
        return

    print(f"Data Loaded: {len(t_sec)} points (Physical Units)")
    
    # 2. Physics Init
    # Duration for Ephemeris
    duration = t_sec[-1].item()
    print(f"Initializing Ephemeris for {duration} seconds...")
    ephemeris = CelestialEphemeris(start_epoch, duration + 1000, step_s=60.0, device=device)
    
    # 3. Model Init
    # We use OrbitalForceNet but maybe we need to scale inputs?
    # R ~ 7000 km. NN likes ~ 1.0.
    # Simple Norm Wrapper
    class NormalizedNN(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.r_scale = 7000.0
            self.v_scale = 7.5
            self.a_scale = 0.001 # km/s^2 (Gravity is ~0.008)
            
        def forward(self, state):
            # Normalize Input
            r = state[:, 0:3] / self.r_scale
            v = state[:, 3:6] / self.v_scale
            x_norm = torch.cat([r, v], dim=1)
            
            # Predict
            out = self.base(x_norm)
            
            # Scale Output
            return out * self.a_scale

    base_net = OrbitalForceNet(input_dim=6, output_dim=3, hidden_dim=256)
    # Warm start if possible?
    prev_weights = os.path.join("weights", "f5_neural_ode_v6.pth")
    if os.path.exists(prev_weights):
        print("Warm starting from v6 weights...")
        try:
            base_net.load_state_dict(torch.load(prev_weights, weights_only=True))
        except:
            print("Weight mismatch or load error. Starting fresh.")
    
    wrapped_net = NormalizedNN(base_net).to(device)
    
    # Dynamics (Physics + NN)
    dynamics = GoldenDynamics(wrapped_net, ephemeris).to(device)
    
    # Optimizers
    # Separate physics params?
    optimizer = optim.AdamW(dynamics.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # 4. Training Loop
    dt = t_sec[1] - t_sec[0] # Assume uniform
    print(f"Integration DT: {dt:.3f} s")
    
    dynamics.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        num_chunks = 15
        
        for _ in range(num_chunks):
            start = np.random.randint(0, len(t_sec) - CHUNK_SIZE - 1)
            end = start + CHUNK_SIZE
            
            target_chunk = states[start:end] #(Chunk, 6)
            t_chunk0 = t_sec[start]
            
            # Integrate
            curr_state = target_chunk[0:1] # (1, 6)
            pred_states = [curr_state]
            
            for i in range(CHUNK_SIZE-1):
                # t relative to start of data?
                # dynamics(t) needs t relative to ephemeris start.
                # t_sec[start + i] is exactly that.
                t_curr = t_sec[start + i]
                
                next_state = rk4_step_golden(dynamics, t_curr, curr_state, dt)
                pred_states.append(next_state)
                curr_state = next_state
                
            pred_stack = torch.cat(pred_states, dim=0)
            
            loss = nn.MSELoss()(pred_stack, target_chunk)
            loss.backward()
            total_loss += loss.item()
            
        optimizer.step()
        scheduler.step(total_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/num_chunks:.8f} | Cr: {torch.exp(dynamics.log_Cr).item():.4f}")
            
    # Save
    torch.save(dynamics.state_dict(), SAVE_PATH)
    print(f"Golden Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_golden()
