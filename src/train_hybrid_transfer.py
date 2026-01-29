import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import datetime
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
from src.models.residual_net import SGP4ErrorCorrector
from src.utils import Normalizer
import requests

# --- Configuration ---
GOLDEN_DATA_PATH = os.path.join("data", "f7_golden_truth.npz")
F5_WEIGHTS_PATH = os.path.join("weights", "hybrid_residual.pth")
F7_WEIGHTS_PATH = os.path.join("weights", "f7_hybrid_transfer.pth")
NORAD_ID = 44387 # Formosat-7

def fetch_tle(norad_id):
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    print(f"Fetching TLE for ID {norad_id}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        l1, l2 = None, None
        for l in lines:
            if l.startswith('1 '): l1 = l
            elif l.startswith('2 '): l2 = l
        return l1, l2
    except:
        return ("1 44387U 19037A   24029.54477817  .00010000  00000-0  17596-3 0  9990",
                "2 44387  24.0000 112.5921 0001222  95.2341 264.9123 15.20023412341234")

def train_hybrid_transfer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training F7 Hybrid Transfer on {device}")
    
    # 1. Load F7 Golden Truth
    data = np.load(GOLDEN_DATA_PATH)
    t_raw = data['t_raw']
    states_golden = data['states_raw']
    start_epoch = float(data['start_epoch'])
    
    # 2. Get F7 SGP4 Reference from Data (ensures precision alignment)
    sgp4_states = data['sgp4_raw']
    normalizer = Normalizer()
    
    # Target: Residual in Position (km)
    # y = R_golden - R_sgp4
    target_res = states_golden[:, 0:3] - sgp4_states[:, 0:3]
    target_res_norm = target_res / normalizer.DU
    
    print(f"Residual Stats - Mean: {np.mean(np.abs(target_res)):.4f} km, Max: {np.max(np.abs(target_res)):.4f} km")
    
    # Input: SGP4 State (Normalized)
    x_train = torch.tensor(normalizer.normalize_state(sgp4_states), dtype=torch.float32).to(device)
    y_train = torch.tensor(target_res_norm, dtype=torch.float32).to(device)
    
    # 3. Define Model (hidden_dim=512 for capacity)
    model = SGP4ErrorCorrector(hidden_dim=512).to(device)
    
    # Load Pre-trained F5 weights
    print(f"Loading F5 pre-trained weights from {F5_WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(F5_WEIGHTS_PATH, weights_only=True))
    
    # Stage 1: Adam for Initial Alignment (Scrubbing bias)
    optimizer_adam = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"Starting Stage 1: Adam Optimization (500 Epochs)...")
    for epoch in range(500):
        optimizer_adam.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer_adam.step()
        
        if (epoch+1) % 100 == 0:
            rms_m = torch.sqrt(loss).item() * normalizer.DU * 1000.0
            print(f"Adam Epoch {epoch+1}/500 | Loss: {loss.item():.10f} | RMS Error: {rms_m:.2f} m")

    # Stage 2: L-BFGS for High-Precision Refinement
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1, max_iter=100, history_size=20, line_search_fn="strong_wolfe")
    
    print(f"Starting Stage 2: L-BFGS Optimization...")
    for epoch in range(20):
        def closure():
            optimizer_lbfgs.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            return loss
        
        optimizer_lbfgs.step(closure)
        with torch.no_grad():
            final_loss = criterion(model(x_train), y_train)
            rms_m = torch.sqrt(final_loss).item() * normalizer.DU * 1000.0
            correction_mag = torch.mean(torch.norm(model(x_train), dim=1)).item() * normalizer.DU
        
        print(f"L-BFGS Epoch {epoch+1}/20 | Loss: {final_loss.item():.10f} | RMS Error: {rms_m:.2f} m | Correction Mag: {correction_mag:.4f} km")
        if rms_m < 5.0: # 5m target
                print("Target precision reached.")
                break
                
    # 4. Save
    os.makedirs(os.path.dirname(F7_WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), F7_WEIGHTS_PATH)
    print(f"F7 Hybrid Transfer weights saved to {F7_WEIGHTS_PATH}")

if __name__ == "__main__":
    train_hybrid_transfer()
