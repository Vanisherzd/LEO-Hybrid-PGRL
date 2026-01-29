import torch
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
from src.models.residual_net import SGP4ErrorCorrector
from src.utils import Normalizer
import requests

# --- Configuration ---
GOLDEN_DATA_PATH = os.path.join("data", "f7_golden_truth.npz")
HYBRID_WEIGHTS = os.path.join("weights", "f7_hybrid_transfer.pth")
PLOT_PATH = os.path.join("plots", "paper", "Fig_PhaseH_F7_Result.png")
NORAD_ID = 44387 # Formosat-7

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
        return ("1 44387U 19037A   24029.54477817  .00010000  00000-0  17596-3 0  9990",
                "2 44387  24.0000 112.5921 0001222  95.2341 264.9123 15.20023412341234")

def evaluate_f7_transfer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Final F7 Hybrid Performance Evaluation on {device}")
    
    # 1. Load F7 Golden Truth
    data = np.load(GOLDEN_DATA_PATH)
    t_raw = data['t_raw']
    states_golden = data['states_raw']
    start_epoch = float(data['start_epoch'])
    
    # 2. SGP4 Prediction from Data (precision alignment)
    sgp4_states = data['sgp4_raw']
    print(f"Loaded {len(sgp4_states)} SGP4 reference points.")
    
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
    
    # 4. Compute Errors
    error_sgp4 = np.linalg.norm(states_golden[:, 0:3] - sgp4_states[:, 0:3], axis=1)
    error_hybrid = np.linalg.norm(states_golden[:, 0:3] - hybrid_r, axis=1)
    
    rmse_sgp4 = np.sqrt(np.mean(error_sgp4**2))
    rmse_hybrid = np.sqrt(np.mean(error_hybrid**2))
    
    print("-" * 30)
    print(f"SGP4 RMSE: {rmse_sgp4:.4f} km")
    print(f"Hybrid Transfer RMSE: {rmse_hybrid:.4f} km")
    print("-" * 30)
    
    if rmse_hybrid < 0.3:
        print(">>> SUCCESS: F7 Hybrid RMSE broke the 300m barrier! <<<")
    else:
        print(">>> Missed 300m target. <<<")
        
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t_raw/60.0, error_sgp4, 'r--', label='SGP4 (Baseline)', alpha=0.7)
    plt.plot(t_raw/60.0, error_hybrid, 'b-', label='Hybrid PGRL (Transferred)', linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Position Error (km)')
    plt.title('Phase H: Formosat-7 Hybrid Transfer Learning Performance')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300)
    print(f"Plot saved to {PLOT_PATH}")
    plt.close()

if __name__ == "__main__":
    evaluate_f7_transfer()
