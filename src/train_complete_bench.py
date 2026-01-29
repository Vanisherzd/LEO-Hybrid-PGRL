import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
from src.models.variants import OrbitalLSTM, OrbitalGRU, OrbitalAttention
from src.models.neural_force import OrbitalForceNet
from src.physics.golden_solver import GoldenDynamics, rk4_step_golden
from src.physics.advanced_forces import CelestialEphemeris
from src.utils import Normalizer

# --- Configuration ---
DATA_PATH = os.path.join("data", "precise_training_data.npz")
WEIGHTS_DIR = "weights"
BENCH_EPOCHS = 50
LR = 1e-3
os.makedirs(WEIGHTS_DIR, exist_ok=True)

class NormalizedNN(nn.Module):
    def __init__(self, base_model, normalizer: Normalizer):
        super().__init__()
        self.base = base_model
        self.register_buffer('scale_r', torch.tensor(normalizer.scale_r, dtype=torch.float32))
        self.register_buffer('scale_v', torch.tensor(normalizer.scale_v, dtype=torch.float32))
        self.register_buffer('scale_a', torch.tensor(normalizer.scale_a, dtype=torch.float32))
        
    def forward(self, state):
        r = state[:, 0:3] / self.scale_r
        v = state[:, 3:6] / self.scale_v
        x_norm = torch.cat([r, v], dim=1) 
        out_norm = self.base(x_norm)
        return out_norm * self.scale_a

def train_one_model(model_name, ModelClass, data_t, data_states, start_epoch, device, force=False):
    weight_path = os.path.join(WEIGHTS_DIR, f"benchmark_{model_name.lower()}.pth")
    if os.path.exists(weight_path) and not force:
        print(f"--- {model_name} weights exist. Skipping... ---")
        return

    print(f"\n=== TRAINING BENCHMARK: {model_name} ===")
    
    # Init Physics
    duration = data_t[-1].item()
    ephemeris = CelestialEphemeris(start_epoch, duration + 1000, step_s=60.0, device=device)
    normalizer = Normalizer()
    dt = data_t[1] - data_t[0]
    
    # Model Setup
    base_net = ModelClass(input_dim=6, output_dim=3, hidden_dim=128)
    wrapped_net = NormalizedNN(base_net, normalizer).to(device)
    dynamics = GoldenDynamics(wrapped_net, ephemeris).to(device)
    
    optimizer = optim.AdamW(dynamics.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BENCH_EPOCHS, eta_min=1e-5)
    
    CHUNK_SIZE = 60
    num_chunks = 5 # Speed priority for benchmark
    
    dynamics.train()
    for epoch in range(BENCH_EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        for _ in range(num_chunks):
            start = np.random.randint(0, len(data_t) - CHUNK_SIZE - 1)
            end = start + CHUNK_SIZE
            target_chunk = data_states[start:end]
            
            curr_state = target_chunk[0:1]
            pred_states = [curr_state]
            
            for i in range(CHUNK_SIZE-1):
                t_curr = data_t[start + i]
                next_state = rk4_step_golden(dynamics, t_curr, curr_state, dt)
                pred_states.append(next_state)
                curr_state = next_state
                
            pred_stack = torch.cat(pred_states, dim=0)
            
            # Dimensionless Loss (Balancing R/V)
            t_r = target_chunk[:, 0:3] / normalizer.scale_r
            t_v = target_chunk[:, 3:6] / normalizer.scale_v
            target_norm = torch.cat([t_r, t_v], dim=1)
            
            p_r = pred_stack[:, 0:3] / normalizer.scale_r
            p_v = pred_stack[:, 3:6] / normalizer.scale_v
            pred_norm = torch.cat([p_r, p_v], dim=1)
            
            loss = nn.MSELoss()(pred_norm, target_norm)
            loss.backward()
            total_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        avg_loss = total_loss / num_chunks
        if (epoch+1) % 10 == 0:
            rmse_est = np.sqrt(avg_loss) * normalizer.scale_r
            print(f"[{model_name}] Ep {epoch+1} | Loss: {avg_loss:.2e} | ~RMSE: {rmse_est:.2f} km")

    # Save
    torch.save(dynamics.state_dict(), weight_path)
    print(f"Saved {model_name} to {weight_path}")

def run_bench():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    
    data = np.load(DATA_PATH)
    t_sec = torch.tensor(data['t_raw'], dtype=torch.float32).to(device)
    states = torch.tensor(data['states_raw'], dtype=torch.float32).to(device)
    start_epoch = float(data['start_epoch']) if 'start_epoch' in data else time.time()
    
    models = [
        ("MLP", OrbitalForceNet, True), 
        ("LSTM", OrbitalLSTM, False),
        ("GRU", OrbitalGRU, False),
        ("Attn", OrbitalAttention, False) 
    ]
    
    for name, cls, force in models:
        train_one_model(name, cls, t_sec, states, start_epoch, device, force=force)

if __name__ == "__main__":
    run_bench()
