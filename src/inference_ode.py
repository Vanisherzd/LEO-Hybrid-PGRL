import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.models.neural_force import OrbitalForceNet
from src.physics.ode_solver import integrate_trajectory
from src.utils import Normalizer

def generate_ode_report():
    # 1. Load Data
    data_path = os.path.join("data", "real_training_data.npz")
    data = np.load(data_path)
    t_norm = data['t']
    states_norm = data['states']
    
    # Validation split (last 20%)
    split_idx = int(0.8 * len(t_norm))
    val_t = t_norm[split_idx:]
    val_states = states_norm[split_idx:] # (Steps, 6)
    
    # Initial Condition for integration
    r0_v0_norm = torch.tensor(val_states[0], dtype=torch.float32).unsqueeze(0) # (1, 6)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference on {device}")
    r0_v0_norm = r0_v0_norm.to(device)
    
    # Load Model
    model = OrbitalForceNet(input_dim=6, output_dim=3, hidden_dim=256).to(device)
    path = os.path.join("weights", "f5_neural_ode_v6.pth")
    if not os.path.exists(path):
        print("Model not found")
        return
        
    print(f"Loading {path}...")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    
    # Integration Params
    total_steps = len(val_t) - 1
    dt_step = (val_t[1] - val_t[0]) # Assuming roughly constant
    
    print(f"Integrating for {total_steps} steps (dt={dt_step:.5f})...")
    
    with torch.no_grad():
        # Integrate full validation trajectory
        # Output: (1, Steps+1, 6)
        pred_traj_norm = integrate_trajectory(r0_v0_norm, float(dt_step), model, steps=total_steps)
        
        # Get learned accelerations for analysis
        # Evaluate model on the trajectory (B, 6) -> (B, 3)
        states_flat = pred_traj_norm.squeeze(0)
        neural_acc = model(states_flat)
        neural_acc_mag = torch.norm(neural_acc, dim=1).cpu().numpy()
        
    pred_traj_norm = pred_traj_norm.squeeze(0).cpu().numpy()
    target_traj_norm = val_states[:total_steps+1] # Match length
    
    # Denormalize
    normalizer = Normalizer()
    preds_km = normalizer.denormalize_state(torch.tensor(pred_traj_norm)).numpy()
    targets_km = normalizer.denormalize_state(torch.tensor(target_traj_norm)).numpy()
    
    # Metrics
    pos_pred = preds_km[:, 0:3]
    pos_target = targets_km[:, 0:3]
    diff = pos_pred - pos_target
    dist_error = np.sqrt(np.sum(diff**2, axis=1))
    
    mean_error = np.mean(dist_error)
    max_error = np.max(dist_error)
    
    print("-" * 60)
    print(f"NEURAL ODE REPORT: Mean Position Error = {mean_error:.4f} km, Max Position Error = {max_error:.4f} km")
    print("-" * 60)
    
    # Plotting
    times_sec = normalizer.denormalize_time(val_t[:len(dist_error)])
    
    plt.figure(figsize=(10, 5))
    plt.plot(times_sec, dist_error, label="Neural ODE Error", color='green')
    plt.xlabel("Time (s)")
    plt.ylabel("Error (km)")
    plt.title(f"Neural ODE Error (Mean: {mean_error:.3f} km)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "neural_ode_error.png"))
    
    # Plot Acceleration Magnitude (Learned Force)
    plt.figure(figsize=(10, 4))
    plt.plot(times_sec, neural_acc_mag, label="|a_neural| (Norm)", color='purple')
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration Magnitude (Dimensionless)")
    plt.title("Learned Perturbation Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "learned_force.png"))
    print("Saved plots.")

if __name__ == "__main__":
    generate_ode_report()
