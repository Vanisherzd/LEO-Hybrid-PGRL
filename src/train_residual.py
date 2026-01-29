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
DATA_PATH = os.path.join("data", "precise_training_data.npz")
WEIGHTS_PATH = os.path.join("weights", "hybrid_residual.pth")
NORAD_ID = 42920

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
        # Fallback to a valid F5 TLE if request fails
        return ("1 42920U 17049A   24029.54477817  .00004500  00000-0  17596-3 0  9990",
                "2 42920  98.2435 112.5921 0001222  95.2341 264.9123 14.50023412341234")

def train_residual():
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
    normalizer = Normalizer()
    
    # Target: Residual in Position (km)
    # y = R_sp3 - R_sgp4
    target_res = states_sp3[:, 0:3] - sgp4_states[:, 0:3]
    
    # Normalizing residuals by DU (Earth Radius) for gradient stability
    target_res_norm = target_res / normalizer.DU
    
    print(f"Residual Stats - Mean: {np.mean(np.abs(target_res)):.2f} km, Max: {np.max(np.abs(target_res)):.2f} km")
    print(f"Normalized Target Stats - Mean: {np.mean(np.abs(target_res_norm)):.4f}")
    
    # Input: SGP4 State (Normalized)
    x_train = torch.tensor(normalizer.normalize_state(sgp4_states), dtype=torch.float32).to(device)
    y_train = torch.tensor(target_res_norm, dtype=torch.float32).to(device)
    
    # 3. Model & Optimizer (Adam then L-BFGS)
    model = SGP4ErrorCorrector(hidden_dim=512).to(device)
    
    # Stage 1: Adam for stable initialization
    optimizer_adam = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"--- Stage 1: Adam Initialization (200 Epochs) ---")
    for epoch in range(200):
        optimizer_adam.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer_adam.step()
        
        if (epoch + 1) % 50 == 0:
            rms_km = torch.sqrt(loss).item() * normalizer.DU
            print(f"Adam Epoch {epoch+1}/200 | Loss: {loss.item():.8f} | RMS: {rms_km:.4f} km")

    # Stage 2: L-BFGS for high-precision refinement
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=100, history_size=20, line_search_fn="strong_wolfe")
    
    print(f"--- Stage 2: L-BFGS Refinement ---")
    
    epochs = 20
    for epoch in range(epochs):
        def closure():
            optimizer_lbfgs.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            return loss
        
        optimizer_lbfgs.step(closure)
        with torch.no_grad():
            final_loss = criterion(model(x_train), y_train)
            rms_km = torch.sqrt(final_loss).item() * normalizer.DU
        
        print(f"L-BFGS Epoch {epoch+1}/{epochs} | Loss: {final_loss.item():.10f} | RMS Error: {rms_km:.4f} km")
        if rms_km < 0.1: # 100m target
                print("Target precision reached.")
                break
                
    # 4. Save
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Hybrid Residual weights saved to {WEIGHTS_PATH}")

if __name__ == "__main__":
    train_residual()
