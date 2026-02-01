import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.pinn.model import HybridPINN
from src.pinn.utils import Normalizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Dataset Wrapper for Inference ---
class InferenceDataset(Dataset):
    def __init__(self, t, states, seq_len=10):
        self.t = torch.tensor(t, dtype=torch.float32).unsqueeze(1)
        self.states = torch.tensor(states, dtype=torch.float32)
        self.seq_len = seq_len
        self.valid_indices = range(seq_len, len(t))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        history = self.states[target_idx - self.seq_len : target_idx]
        target_state = self.states[target_idx]
        target_t = self.t[target_idx]
        return target_t, history, target_state, target_idx

def generate_report():
    # 1. Load Data
    data_path = os.path.join("data", "real_training_data.npz")
    if not os.path.exists(data_path):
        print("Data not found.")
        return

    data = np.load(data_path)
    t_norm = data['t']
    states_norm = data['states']
    
    # We'll use the validation set (last 20%)
    split_idx = int(0.8 * len(t_norm))
    val_t = t_norm[split_idx:]
    val_states = states_norm[split_idx:]
    
    val_dataset = InferenceDataset(val_t, val_states)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # 2. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference on {device}")
    
    model = HybridPINN(input_dim=6, hidden_dim=128).to(device)
    
    model_path = os.path.join("weights", "f5_realdata_v1.pth")
    # if not os.path.exists(model_path):
    #     print(f"Gentle model {model_path} not found, checking v4...")
    #     model_path = "f5_precision_v4.pth"
        
    print(f"Loading model from: {model_path}")
    # strict=False allows loading v1/v2/v3 weights into v4 code (with new log_ballistic_coeff)
    # The new parameter will stay at its initialization (0.0) if missing in file.
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    
    # 3. Inference
    all_preds_norm = []
    all_targets_norm = []
    all_times_norm = []
    
    print("Running Inference...")
    with torch.no_grad():
        for t_batch, hist_batch, target_batch, idx_batch in val_loader:
            t_batch = t_batch.to(device)
            hist_batch = hist_batch.to(device)
            
            pred = model(t_batch, hist_batch)
            
            all_preds_norm.append(pred.cpu().numpy())
            all_targets_norm.append(target_batch.cpu().numpy())
            all_times_norm.append(t_batch.cpu().numpy())
            
    all_preds_norm = np.concatenate(all_preds_norm, axis=0)
    all_targets_norm = np.concatenate(all_targets_norm, axis=0) # (N, 6)
    all_times_norm = np.concatenate(all_times_norm, axis=0) # (N, 1)

    # 4. Denormalize
    normalizer = Normalizer()
    
    # preds: (N, 6) -> split to r and v
    preds_km = normalizer.denormalize_state(torch.tensor(all_preds_norm))
    targets_km = normalizer.denormalize_state(torch.tensor(all_targets_norm))
    
    # Convert back to numpy for metrics
    preds_km = preds_km.numpy()
    targets_km = targets_km.numpy()
    
    # Times (relative seconds)
    times_sec = normalizer.denormalize_time(all_times_norm)

    # 5. Metrics Calculation (Position Error)
    # Position columns: 0,1,2
    pos_pred = preds_km[:, 0:3]
    pos_target = targets_km[:, 0:3]
    
    # Euclidean Distance: sqrt(dx^2 + dy^2 + dz^2)
    diff = pos_pred - pos_target
    dist_error = np.sqrt(np.sum(diff**2, axis=1)) # (N,)
    
    mean_error = np.mean(dist_error)
    max_error = np.max(dist_error)
    
    print("-" * 60)
    print(f"EVALUATION REPORT: Mean Position Error = {mean_error:.4f} km, Max Position Error = {max_error:.4f} km")
    print("-" * 60)
    
    # 6. Plotting
    # A. Error Plot
    plt.figure(figsize=(10, 5))
    plt.plot(times_sec, dist_error, label="Position Error", color='red')
    plt.xlabel("Time (seconds from start of test)")
    plt.ylabel("Error (km)")
    plt.title(f"Prediction Error over Time (Mean: {mean_error:.3f} km)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "prediction_error.png"))
    print("Saved prediction_error.png")
    
    # B. Trajectory Comparison (XY Plane)
    plt.figure(figsize=(8, 8))
    plt.plot(pos_target[:, 0], pos_target[:, 1], 'k--', label="Ground Truth (SGP4)", alpha=0.7)
    plt.plot(pos_pred[:, 0], pos_pred[:, 1], 'b-', label="HybridPINN Prediction", alpha=0.6)
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.title("Trajectory Comparison (XY Plane)")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "trajectory_compare.png"))
    print("Saved trajectory_compare.png")

if __name__ == "__main__":
    generate_report()
