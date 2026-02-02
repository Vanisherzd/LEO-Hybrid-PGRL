import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import argparse

# Force relative imports to work
sys.path.append(os.getcwd())

from src.pinn.train_complete_bench import train_one_model
from src.models.variants import OrbitalLSTM, OrbitalGRU, OrbitalAttention
from src.models.neural_force import OrbitalForceNet
from src.pinn.train_residual import train_residual
from src.rl.agent_train import train_rl_agent

# Standard Paths
LOGS_DIR = "logs"
WEIGHTS_DIR = "weights"
MODELS_DIR = "models/rl"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def run_full_pipeline(epochs_bench=50, adam_hybrid=200, lbfgs_hybrid=20, steps_rl=50000):
    print("\n" + "="*50)
    print("🚀 STARTING GRAND REPRODUCTION SUITE")
    print(f"Mode Params: Bench={epochs_bench}, Hybrid(A/L)={adam_hybrid}/{lbfgs_hybrid}, RL={steps_rl}")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    
    # ---------------------------------------------------------
    # TASK A: Benchmark Architectures
    # ---------------------------------------------------------
    print("\n--- [TASK A] Training Benchmark Architectures ---")
    data_bench = np.load(os.path.join("data", "precise_training_data.npz"))
    t_bench = torch.tensor(data_bench['t_raw'], dtype=torch.float32).to(device)
    states_bench = torch.tensor(data_bench['states_raw'], dtype=torch.float32).to(device)
    start_epoch = float(data_bench['start_epoch'])
    
    bench_results = []
    models_to_test = [
        ("MLP", OrbitalForceNet), 
        ("LSTM", OrbitalLSTM),
        ("GRU", OrbitalGRU),
        ("Attn", OrbitalAttention) 
    ]
    
    for name, cls in models_to_test:
        print(f"\n[Bench] Starting {name} training...")
        start_time = time.time()
        final_rmse = train_one_model(name, cls, t_bench, states_bench, start_epoch, device, force=True, epochs=epochs_bench)
        end_time = time.time()
        
        bench_results.append({
            "model": name,
            "train_time_s": end_time - start_time,
            "final_rmse": float(final_rmse),
            "status": "Success"
        })
    
    pd.DataFrame(bench_results).to_csv(os.path.join(LOGS_DIR, "bench_metrics.csv"), index=False)
    print(f"Benchmark metrics saved to {LOGS_DIR}/bench_metrics.csv")

    # ---------------------------------------------------------
    # TASK B: Hybrid PGRL (F5)
    # ---------------------------------------------------------
    print("\n--- [TASK B] Training Hybrid PGRL (SGP4 Corrector) ---")
    train_residual(adam_epochs=adam_hybrid, lbfgs_epochs=lbfgs_hybrid)
    
    # Rename for consistency with directive
    if os.path.exists("weights/hybrid_residual.pth"):
        os.replace("weights/hybrid_residual.pth", "weights/hybrid_f5.pth")
    print("Hybrid PGRL training complete. Weights: weights/hybrid_f5.pth")

    # ---------------------------------------------------------
    # TASK C: RL PPO Agent
    # ---------------------------------------------------------
    print("\n--- [TASK C] Training RL PPO Agent ---")
    train_rl_agent(total_timesteps=steps_rl)
    
    # Ensure naming matches Fig4 expectations
    if os.path.exists("logs/rl_training_history.csv"):
        os.replace("logs/rl_training_history.csv", "logs/rl_metrics.csv")
    
    print("\n" + "="*50)
    print("✅ REPRODUCTION SUITE COMPLETE")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEO-Hybrid-PGRL Master Reproduction")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "paper"],
                        help="Quick (Low epochs) or Paper (full training)")
    args = parser.parse_args()
    
    if args.mode == "quick":
        # Fast validation run
        run_full_pipeline(epochs_bench=2, adam_hybrid=10, lbfgs_hybrid=2, steps_rl=2048)
    else:
        # Full scientific run
        run_full_pipeline(epochs_bench=100, adam_hybrid=500, lbfgs_hybrid=50, steps_rl=100000)
