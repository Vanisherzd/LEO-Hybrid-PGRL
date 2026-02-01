import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from src.rl.env import LEOCommEnv

# Paths
WEIGHTS_DIR = "weights"
PLOTS_DIR = os.path.join("plots", "rl")
os.makedirs(PLOTS_DIR, exist_ok=True)
RL_MODEL_PATH = os.path.join(WEIGHTS_DIR, "rl_policy_mac.zip")

def evaluate_rl_vs_baseline():
    print("--- Phase I: Academic Benchmarking (RL vs Fixed Baseline) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = LEOCommEnv(device=device)
    
    # 1. Load RL Model
    if not os.path.exists(RL_MODEL_PATH):
        print(f"Error: Trained model not found at {RL_MODEL_PATH}. Run training first.")
        return
    model = PPO.load(RL_MODEL_PATH, device=device)
    
    # 2. Results Containers
    results = {
        "RL": {"times": [], "actions": [], "rewards": [], "power": [], "throughput": [], "packet_loss": []},
        "Fixed_Low": {"times": [], "actions": [], "rewards": [], "power": [], "throughput": [], "packet_loss": []},
        "Fixed_High": {"times": [], "actions": [], "rewards": [], "power": [], "throughput": [], "packet_loss": []}
    }
    
    # 3. Evaluation Loops
    for label in results.keys():
        print(f"Evaluating {label}...")
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            # Select Action
            if label == "RL":
                action, _ = model.predict(obs, deterministic=True)
            elif label == "Fixed_Low":
                action = np.array([0.1]) # Fixed 10% Guard Band
            elif label == "Fixed_High":
                action = np.array([0.4]) # Fixed 40% Guard Band
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record Metrics
            results[label]["times"].append(step * 10.0 / 60.0) # minutes
            results[label]["actions"].append(action[0])
            results[label]["rewards"].append(reward)
            results[label]["power"].append(info["power_consumed_j"])
            results[label]["throughput"].append(info["data_rate_bps"] / 1e3) # kbps
            results[label]["packet_loss"].append(1.0 if not info["success"] else 0.0)
            
            step += 1
            
    # 4. Visualization: Metric 1 - Power vs Throughput (Pareto Frontier)
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        avg_power = np.mean(data["power"]) * 1000 # mJ
        avg_throughput = np.mean(data["throughput"])
        color = 'blue' if label == "RL" else 'gray'
        marker = 'o' if label == "RL" else 'x'
        plt.scatter(avg_power, avg_throughput, label=f"{label}", color=color, s=200, marker=marker, edgecolors='black')
        plt.text(avg_power + 1, avg_throughput + 1, label, fontsize=12, fontweight='bold')

    plt.xlabel("Average Power Consumption (mJ)")
    plt.ylabel("Average Throughput (kbps)")
    plt.title("Metric 1: Power-Throughput Efficiency (Pareto Frontier)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "Fig_RL_Metric1_Pareto.png"))

    # 5. Visualization: Metric 2 - Dynamic Guard Band Allocation
    plt.figure(figsize=(10, 6))
    plt.plot(results["RL"]["times"], results["RL"]["actions"], label="Autonomous RL Action", color='blue', linewidth=2)
    plt.axhline(y=0.1, color='green', linestyle='--', label="Fixed Baseline (Low)")
    plt.axhline(y=0.4, color='red', linestyle='--', label="Fixed Baseline (High)")
    plt.xlabel("Time in Contact (minutes)")
    plt.ylabel("Guard Band Ratio")
    plt.title("Metric 2: Dynamic Guard Band Allocation (RL Adaption)")
    plt.ylim(0, 0.6)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "Fig_RL_Metric2_Adaption.png"))

    # 6. Visualization: Metric 3 - QoS (Packet Loss) Accumulation
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        loss_rate = np.cumsum(data["packet_loss"]) / (np.arange(len(data["packet_loss"])) + 1)
        plt.plot(data["times"], loss_rate, label=label)
        
    plt.xlabel("Time (minutes)")
    plt.ylabel("Moving Average Packet Loss Rate")
    plt.title("Metric 3: QoS Optimization (Reliability)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR, "Fig_RL_Metric3_QoS.png"))

    print(f"All academic plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    evaluate_rl_vs_baseline()
