import os
import sys
import matplotlib
matplotlib.use('Agg') # Headless backend
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sgp4.api import Satrec, WGS72, jday

# Force relative imports
sys.path.append(os.getcwd())

from src.pinn.utils import Normalizer
from src.models.residual_net import SGP4ErrorCorrector
from src.models.neural_force import OrbitalForceNet
from src.models.variants import OrbitalLSTM, OrbitalGRU, OrbitalAttention
from src.physics.golden_solver import GoldenDynamics, rk4_step_golden
from src.physics.advanced_forces import CelestialEphemeris
from src.utils.plot_styles import set_academic_style, get_color_palette

# Set Academic Style
set_academic_style()
colors = get_color_palette()

# Paths
DATA_PATH = os.path.join("data", "precise_training_data.npz")
WEIGHTS_DIR = "weights"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def compute_ric_error(r_true, v_true, r_pred):
    """
    Transforms position error into RIC (Radial, Intrack, Crosstrack) frame.
    """
    # Radial unit vector
    u_r = r_true / np.linalg.norm(r_true)
    # Crosstrack unit vector (Normal to orbital plane)
    u_c = np.cross(r_true, v_true)
    u_c = u_c / np.linalg.norm(u_c)
    # Intrack unit vector
    u_i = np.cross(u_c, u_r)
    
    delta_r = r_pred - r_true
    
    err_r = np.dot(delta_r, u_r)
    err_i = np.dot(delta_r, u_i)
    err_c = np.dot(delta_r, u_c)
    
    return err_r, err_i, err_c

class BenchmarkWrapper(torch.nn.Module):
    def __init__(self, base_model, normalizer):
        super().__init__()
        self.base = base_model
        self.register_buffer('scale_r', torch.tensor(normalizer.scale_r, dtype=torch.float32))
        self.register_buffer('scale_v', torch.tensor(normalizer.scale_v, dtype=torch.float32))
        self.register_buffer('scale_a', torch.tensor(normalizer.scale_a, dtype=torch.float32))
        
    def forward(self, state):
        # state is (Batch, 6) in km, km/s
        r_norm = state[:, 0:3] / self.scale_r
        v_norm = state[:, 3:6] / self.scale_v
        x_norm = torch.cat([r_norm, v_norm], dim=1)
        a_norm = self.base(x_norm)
        return a_norm * self.scale_a

def run_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalizer = Normalizer()
    
    # 1. Load Truth Data
    print("Loading data...")
    data = np.load(DATA_PATH)
    t_raw = data['t_raw'][:120] # 120 minutes (assuming 60s steps)
    states_true = data['states_raw'][:120]
    start_epoch = float(data['start_epoch'])
    dt = 60.0
    
    # 2. Setup Models
    print("Initializing models...")
    models = {}
    
    # SGP4 Setup (Latest 2026 TLE)
    l1 = "1 42920U 17049A   26033.40794503  .00010884  00000-0  34483-3 0  9998"
    l2 = "2 42920  98.2443  18.5369 0001859  38.7495 321.3653 14.50153835359281"
    sat = Satrec.twoline2rv(l1, l2, WGS72)
    start_dt = datetime.datetime.fromtimestamp(start_epoch, tz=datetime.timezone.utc)
    
    # AI Models
    architectures = [
        ("MLP", OrbitalForceNet, "benchmark_mlp.pth"),
        ("LSTM", OrbitalLSTM, "benchmark_lstm.pth"),
        ("GRU", OrbitalGRU, "benchmark_gru.pth"),
        ("Attn", OrbitalAttention, "benchmark_attn.pth")
    ]
    
    # Ephemeris for ODE models
    ephem = CelestialEphemeris(start_epoch, t_raw[-1]+100, step_s=60.0, device=device)
    
    for name, cls, w_name in architectures:
        w_path = os.path.join(WEIGHTS_DIR, w_name)
        if os.path.exists(w_path):
            net = cls(input_dim=6, output_dim=3, hidden_dim=128).to(device)
            wrapper = BenchmarkWrapper(net, normalizer)
            dyn = GoldenDynamics(wrapper, ephem).to(device)
            # Benchmark weights were saved as dyn.state_dict()
            dyn.load_state_dict(torch.load(w_path, weights_only=True))
            dyn.eval()
            models[name] = dyn
            
    # Hybrid Model
    hybrid_path = os.path.join(WEIGHTS_DIR, "hybrid_f5.pth")
    if os.path.exists(hybrid_path):
        h_net = SGP4ErrorCorrector(hidden_dim=512).to(device)
        h_net.load_state_dict(torch.load(hybrid_path, weights_only=True))
        h_net.eval()
        models["Hybrid"] = h_net
        
    print(f"Models loaded: {list(models.keys())} + SGP4")
    
    results = {name: [] for name in list(models.keys()) + ["SGP4"]}
    ric_hybrid = [] # To store (r, i, c) for Hybrid
    
    # 3. Integration Loop
    # All models start from the EXACT same state
    # 1. Aligned SGP4 Baseline
    # We compute the initial SGP4 state to calculate the bias/offset at t=0
    jd_0, fr_0 = jday(start_dt.year, start_dt.month, start_dt.day, 
                      start_dt.hour, start_dt.minute, start_dt.second)
    _, r0_raw, v0_raw = sat.sgp4(jd_0, fr_0)
    
    # Biases to ensure exactly 0.0 at t=0
    sgp4_bias = np.array(r0_raw) - states_true[0, 0:3]
    hybrid_bias = None
    
    # ODE initial states
    ode_states = {name: torch.tensor(states_true[0:1], dtype=torch.float32).to(device) 
                  for name in models if name not in ["Hybrid", "SGP4"]}
    
    print("\nStarting 120-minute head-to-head integration...")
    for i, t_val in enumerate(t_raw):
        # Current SGP4 Raw
        curr_dt = start_dt + datetime.timedelta(seconds=float(t_val))
        jd, fr = jday(curr_dt.year, curr_dt.month, curr_dt.day, 
                      curr_dt.hour, curr_dt.minute, curr_dt.second)
        _, r_raw, v_raw = sat.sgp4(jd, fr)
        r_raw = np.array(r_raw)
        
        # 1. Aligned SGP4 Baseline (The Physics Anchor)
        r_s_aligned = r_raw - sgp4_bias
        results["SGP4"].append(np.linalg.norm(r_s_aligned - states_true[i, 0:3]))
        
        # 2. Hybrid PGRL (Corrections applied to the Aligned Anchor)
        if "Hybrid" in models:
            s_raw_vec = np.concatenate([r_raw, v_raw]) # NN was trained on RAW inputs
            x_norm = torch.tensor(normalizer.normalize_state(s_raw_vec.reshape(1, 6)), dtype=torch.float32).to(device)
            with torch.no_grad():
                dr_norm = models["Hybrid"](x_norm).cpu().numpy()[0]
            
            # Prediction: Aligned Anchor + Neural Correction
            r_hyb = r_s_aligned + (dr_norm * normalizer.DU)
            
            # Ensure relative zero at t=0 if there's any residual NN bias
            if i == 0:
                h_init_error = r_hyb - states_true[0, 0:3]
                print(f"Alignment Sync - SGP4 Init Error: {results['SGP4'][0]:.6f} km")
                print(f"Alignment Sync - Hybrid Init Error (pre-bias): {np.linalg.norm(h_init_error):.6f} km")
            
            r_hyb_final = r_hyb - h_init_error
            results["Hybrid"].append(np.linalg.norm(r_hyb_final - states_true[i, 0:3]))
            ric_hybrid.append(compute_ric_error(states_true[i, 0:3], states_true[i, 3:6], r_hyb_final))
            
        # Neural ODEs (Independent Integration)
        for name in ode_states:
            if i > 0:
                with torch.no_grad():
                    ode_states[name] = rk4_step_golden(models[name], torch.tensor(t_raw[i-1]).to(device), ode_states[name], dt)
            err = np.linalg.norm(ode_states[name].cpu().numpy()[0, 0:3] - states_true[i, 0:3])
            results[name].append(err)
            
    # 4. Visualization
    time_min = t_raw / 60.0
    
    # Plot A: Global Trajectory Error
    plt.figure(figsize=(12, 7))
    plt.plot(time_min, results["SGP4"], color='black', linestyle='--', label='SGP4 Baseline', alpha=0.7)
    if "Hybrid" in results:
        plt.plot(time_min, results["Hybrid"], color='red', linewidth=3, label='Hybrid PGRL (Ours)')
    
    ai_colors = ['blue', 'cyan', 'green', 'magenta']
    ai_idx = 0
    for name in results:
        if name not in ["SGP4", "Hybrid"]:
            plt.plot(time_min, results[name], color=ai_colors[ai_idx % len(ai_colors)], label=f'Pure {name}', alpha=0.6)
            ai_idx += 1
            
    plt.yscale('log')
    plt.ylim(1e-2, 1e4) # Unified scale from user request
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Position Error (km)")
    plt.title("Fig A: Global Trajectory Error Head-to-Head (120-min)")
    plt.legend(loc='upper left', ncol=2)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(PLOTS_DIR, "FigA_Global_Trajectory_Error.png"), dpi=300)
    print("Saved FigA.")
    
    # Plot B: Final RMSE Bar Chart
    plt.figure(figsize=(10, 6))
    final_errs = {name: results[name][-1] for name in results}
    names = list(final_errs.keys())
    vals = list(final_errs.values())
    
    bars = plt.bar(names, vals, color=['black', 'red'] + ai_colors[:len(names)-2])
    plt.yscale('log')
    plt.ylabel("Final RMSE at 120min (km)")
    plt.title("Fig B: Final Error Comparison (End-of-Life State)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "FigB_Final_RMSE_Benchmark.png"), dpi=300)
    print("Saved FigB.")
    
    # Plot C: RIC Breakdown (Hybrid)
    if ric_hybrid:
        ric_arr = np.array(ric_hybrid)
        plt.figure(figsize=(10, 6))
        plt.plot(time_min, ric_arr[:, 0], label='Radial', color='orange')
        plt.plot(time_min, ric_arr[:, 1], label='Intrack (Drag Direction)', color='green')
        plt.plot(time_min, ric_arr[:, 2], label='Crosstrack', color='blue')
        plt.xlabel("Time (Minutes)")
        plt.ylabel("Residual Error (km)")
        plt.title("Fig C: RIC Error Breakdown (Hybrid Model)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOTS_DIR, "FigC_RIC_Breakdown.png"), dpi=300)
        print("Saved Fig C.")

    # 5. Result Table Output
    print("\n" + "="*50)
    print("UNIFIED VALIDATION FINAL RESULTS")
    print("="*50)
    print("| Model Architecture | Final RMSE (120m) | Stability Status |")
    print("| :--- | :--- | :--- |")
    for name in results:
        rmse = results[name][-1]
        status = "Convergent" if rmse < 1.0 else ("Stable" if rmse < 5.0 else "DIVERGENT")
        print(f"| {name} | {rmse:.4f} km | {status} |")
    print("="*50)

if __name__ == "__main__":
    run_validation()
