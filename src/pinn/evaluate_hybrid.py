import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import time
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
from src.models.residual_net import SGP4ErrorCorrector
from src.models.neural_force import OrbitalForceNet
from src.physics.golden_solver import GoldenDynamics, rk4_step_golden
from src.physics.advanced_forces import CelestialEphemeris
from src.pinn.utils import Normalizer
import requests

# --- Configuration ---
DATA_PATH = os.path.join("data", "precise_training_data.npz")
HYBRID_WEIGHTS = os.path.join("weights", "hybrid_residual.pth")
MLP_WEIGHTS = os.path.join("weights", "benchmark_mlp.pth")
PLOTS_DIR = os.path.join("plots", "paper")
os.makedirs(PLOTS_DIR, exist_ok=True)
NORAD_ID = 42920

def fetch_tle(norad_id):
    try:
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
        response = requests.get(url, timeout=10)
        lines = response.text.strip().splitlines()
        l1, l2 = None, None
        for l in lines:
            if l.startswith('1 '): l1 = l
            elif l.startswith('2 '): l2 = l
        return l1, l2
    except:
        return ("1 42920U 17049A   24029.54477817  .00004500  00000-0  17596-3 0  9990",
                "2 42920  98.2435 112.5921 0001222  95.2341 264.9123 14.50023412341234")

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

def evaluate_hybrid():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hybrid Performance Evaluation on {device}")
    
    # 1. Load Data
    data = np.load(DATA_PATH)
    t_raw = data['t_raw']
    states_sp3 = data['states_raw']
    start_epoch = float(data['start_epoch'])
    
    # 2. SGP4 Reference
    l1, l2 = fetch_tle(NORAD_ID)
    satellite = Satrec.twoline2rv(l1, l2, WGS72)
    start_dt = datetime.datetime.fromtimestamp(start_epoch, tz=datetime.timezone.utc)
    
    sgp4_states = []
    for t_sec in t_raw:
        curr_dt = start_dt + datetime.timedelta(seconds=float(t_sec))
        jd, fr = jday(curr_dt.year, curr_dt.month, curr_dt.day, 
                      curr_dt.hour, curr_dt.minute, curr_dt.second)
        e, r, v = satellite.sgp4(jd, fr)
        sgp4_states.append(list(r) + list(v))
    sgp4_states = np.array(sgp4_states)
    
    # 3. Hybrid Prediction
    normalizer = Normalizer()
    hybrid_model = SGP4ErrorCorrector(hidden_dim=512).to(device)
    hybrid_model.load_state_dict(torch.load(HYBRID_WEIGHTS, weights_only=True))
    hybrid_model.eval()
    
    x_input = torch.tensor(normalizer.normalize_state(sgp4_states), dtype=torch.float32).to(device)
    with torch.no_grad():
        correction_norm = hybrid_model(x_input).cpu().numpy()
        correction = correction_norm * normalizer.DU
        
    hybrid_r = sgp4_states[:, 0:3] + correction
    
    print("\n--- Debug Residual Check (Point 0) ---")
    print(f"SGP4 R: {sgp4_states[0, 0:3]}")
    print(f"SP3 R: {states_sp3[0, :3]}")
    print(f"Predicted Correction: {correction[0]}")
    print(f"True Residual: {states_sp3[0, :3] - sgp4_states[0, 0:3]}")
    print(f"Hybrid R: {hybrid_r[0]}")
    print(f"Error SGP4: {np.linalg.norm(states_sp3[0, :3] - sgp4_states[0, 0:3]):.4f} km")
    print(f"Error Hybrid: {np.linalg.norm(states_sp3[0, :3] - hybrid_r[0]):.4f} km")
    
    # 4. Neural ODE (MLP) Prediction (For Comparison)
    mlp_base = OrbitalForceNet(input_dim=6, output_dim=3, hidden_dim=128)
    wrapped = NormalizedNN(mlp_base, normalizer).to(device)
    ephemeris = CelestialEphemeris(start_epoch, t_raw[-1] + 1000, device=device)
    dynamics = GoldenDynamics(wrapped, ephemeris).to(device)
    dynamics.load_state_dict(torch.load(MLP_WEIGHTS, weights_only=True), strict=False)
    dynamics.eval()
    
    # Integrate for 100 points to show divergence
    eval_len = 100 
    curr_state = torch.tensor(states_sp3[0:1], dtype=torch.float32).to(device)
    dt = t_raw[1] - t_raw[0]
    ode_states = [curr_state.cpu().numpy()[0]]
    with torch.no_grad():
        for i in range(eval_len - 1):
            curr_state = rk4_step_golden(dynamics, t_raw[i], curr_state, dt)
            ode_states.append(curr_state.cpu().numpy()[0])
    ode_states = np.array(ode_states)
    
    # 5. Compute Errors
    sgp4_err = np.linalg.norm(sgp4_states[:, 0:3] - states_sp3[:, 0:3], axis=1)
    hybrid_err = np.linalg.norm(hybrid_r - states_sp3[:, 0:3], axis=1)
    ode_err = np.linalg.norm(ode_states[:, 0:3] - states_sp3[:eval_len, 0:3], axis=1)
    
    # 6. Plotting
    t_mins = t_raw / 60.0
    plt.figure(figsize=(10, 6))
    plt.plot(t_mins, sgp4_err, 'r--', label='SGP4 (Oscillating)', alpha=0.7)
    plt.plot(t_mins[:eval_len], ode_err, 'g:', label='Neural ODE (Diverging)', linewidth=2)
    plt.plot(t_mins, hybrid_err, 'b-', label='Hybrid PGRL (SGP4 + Residual NN)', linewidth=2.5)
    
    plt.yscale('log')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Position Error (km)')
    plt.title('Fig: Hybrid Optimization Performance (PGRL Strategy)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(PLOTS_DIR, "Fig_Hybrid_Performance.png")
    plt.savefig(plot_path)
    print(f"Saved hybrid performance plot to {plot_path}")
    
    print(f"\n--- Final Metrics ---")
    print(f"SGP4 Average Error: {np.mean(sgp4_err):.4f} km")
    print(f"Hybrid PGRL Average Error: {np.mean(hybrid_err):.4f} km")
    print(f"Target Threshold: 0.5 km")
    if np.mean(hybrid_err) < 0.5:
        print("SUCCESS: Hybrid approach achieved sub-500m accuracy!")
    else:
        print("STAGNATION: Hybrid approach improved error but missed sub-500m target.")

if __name__ == "__main__":
    evaluate_hybrid()
