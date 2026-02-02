import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import datetime
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
from src.models.residual_net import SGP4ErrorCorrector
from src.pinn.utils import Normalizer
import requests

# --- Configuration ---
DATA_PATH = os.path.join("data", "precise_training_data.npz")
WEIGHTS_PATH = os.path.join("weights", "hybrid_residual.pth")
NORAD_ID = 42920

def fetch_tle(norad_id):
    # HARDCODED FOR ABSOLUTE SYNC
    return ("1 42920U 17049A   26033.40794503  .00010884  00000-0  34483-3 0  9998",
            "2 42920  98.2443  18.5369 0001859  38.7495 321.3653 14.50153835359281")

def train_residual(adam_epochs=200, lbfgs_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Residual Model on {device}")
    
    # 1. Load SP3 Data
    data = np.load(DATA_PATH)
    t_raw = data['t_raw'] # seconds from start_epoch
    states_sp3 = data['states_raw'] # (N, 6) [r, v] km, km/s
    start_epoch = float(data['start_epoch'])
    
    # 2. Get SGP4 Trayectory
    l1, l2 = fetch_tle(NORAD_ID)
    satellite = Satrec.twoline2rv(l1, l2, WGS72)
    
    start_dt = datetime.datetime.fromtimestamp(start_epoch, tz=datetime.timezone.utc)
    
    sgp4_states = []
    print(f"Computing SGP4 residuals for {len(t_raw)} points...")
    for t_sec in t_raw:
        curr_dt = start_dt + datetime.timedelta(seconds=float(t_sec))
        jd, fr = jday(curr_dt.year, curr_dt.month, curr_dt.day, 
                      curr_dt.hour, curr_dt.minute, curr_dt.second)
        e, r, v = satellite.sgp4(jd, fr)
        sgp4_states.append(list(r) + list(v))
    
    sgp4_states = np.array(sgp4_states)
    
    # 2.5 Align SGP4 to SP3 at t=0
    # This prevents the NN from trying to learn geometric frame rotation (TEME vs ECI)
    # and focuses it on physical drift.
    sgp4_bias = sgp4_states[0, 0:3] - states_sp3[0, 0:3]
    sgp4_states_aligned = sgp4_states.copy()
    sgp4_states_aligned[:, 0:3] -= sgp4_bias
    
    normalizer = Normalizer()
    
    # Target: Residual in Position (km) relative to ALIGNED SGP4
    target_res = states_sp3[:, 0:3] - sgp4_states_aligned[:, 0:3]
    
    # RESIDUAL SCALING: Amplify residuals to O(1) for optimizer stability
    # If residual is ~1km, target is 1.0. If 1m, target is 0.001.
    # User Request: "Ensure the Normalizer amplifies these residuals"
    # We will scale by a factor of 100.0 (so 10m = 0.001 -> 1.0) 
    # RES_SCALE = 5000.0 (Synchronized with F7 for sub-300m target)
    RES_SCALE = 5000.0
    target_res_norm = target_res * RES_SCALE
    
    print(f"Residual Stats - Mean: {np.mean(np.abs(target_res)):.2f} km, Max: {np.max(np.abs(target_res)):.2f} km")
    print(f"Normalized Target Stats - Mean: {np.mean(np.abs(target_res_norm)):.4f}")
    
    # Input: SGP4 State (Normalized)
    x_train = torch.tensor(normalizer.normalize_state(sgp4_states), dtype=torch.float32).to(device)
    y_train = torch.tensor(target_res_norm, dtype=torch.float32).to(device)
    
    # 3. Model & Optimizer (Adam then L-BFGS)
    model = SGP4ErrorCorrector(hidden_dim=256).to(device)
    
    # Stage 1: Adam for stable initialization
    optimizer_adam = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"--- Stage 1: Adam Initialization ({adam_epochs} Epochs) ---")
    for epoch in range(adam_epochs):
        optimizer_adam.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer_adam.step()
        
        if (epoch + 1) % 50 == 0:
            rms_km = torch.sqrt(loss).item() / RES_SCALE
            print(f"Adam Epoch {epoch+1}/{adam_epochs} | Loss: {loss.item():.8f} | RMS: {rms_km:.6f} km")

    # Stage 2: L-BFGS for high-precision refinement
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=100, history_size=50, line_search_fn="strong_wolfe")
    
    print(f"--- Stage 2: L-BFGS Refinement ---")
    
    def closure():
        optimizer_lbfgs.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        return loss

    for epoch in range(lbfgs_epochs):
        optimizer_lbfgs.step(closure)
        with torch.no_grad():
            final_loss = criterion(model(x_train), y_train)
            rms_km = torch.sqrt(final_loss).item() / RES_SCALE
        
        print(f"L-BFGS Epoch {epoch+1}/{lbfgs_epochs} | Loss: {final_loss.item():.10f} | RMS Error: {rms_km:.6f} km")
        if rms_km < 0.0001: # 10cm target
                print("Target precision reached.")
                break
                
    # 4. Save
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Hybrid Residual weights saved to {WEIGHTS_PATH}")

if __name__ == "__main__":
    train_residual()
