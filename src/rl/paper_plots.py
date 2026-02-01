import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from src.rl.env import LEOCommEnv

# Settings
PLOTS_DIR = os.path.join("plots", "rl")
LOGS_DIR = "logs"
MODEL_PATH = os.path.join("models", "rl", "rl_policy_mac.zip")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 10,
    "grid.alpha": 0.3
})

def generate_plot_a_learning_curve():
    print("Generating Plot A: Learning Curve...")
    csv_path = os.path.join(LOGS_DIR, "rl_training_history.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping Plot A.")
        return
        
    df = pd.read_csv(csv_path)
    # Smoothing
    df['reward_smooth'] = df['reward'].rolling(window=10).mean()
    
    plt.figure(figsize=(8, 5))
    plt.plot(df['step'], df['reward'], alpha=0.3, color='blue', label='Episode Reward')
    plt.plot(df['step'], df['reward_smooth'], color='blue', linewidth=2, label='Moving Average (10 episodes)')
    
    plt.xlabel("Total Timesteps")
    plt.ylabel("Reward per Episode")
    plt.title("A. RL Training Convergence (PPO)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "Paper_PlotA_LearningCurve.png"), dpi=300)
    print("Saved Plot A.")

def generate_plot_b_pareto():
    print("Generating Plot B: Pareto Frontier...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = LEOCommEnv(device=device)
    model = PPO.load(MODEL_PATH, device=device)
    
    # Baselines
    baselines = {
        "RL Agent (Autonomous)": "optimal",
        "Fixed MAC (Low 5%)": 0.05,
        "Fixed MAC (Med 20%)": 0.20,
        "Fixed MAC (High 45%)": 0.45
    }
    
    data_points = []
    
    for label, strategy in baselines.items():
        obs, _ = env.reset()
        total_pwr = 0
        total_data = 0
        steps = 0
        done = False
        
        while not done:
            if strategy == "optimal":
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.array([strategy])
                
            obs, reward, term, trunc, info = env.step(action)
            total_pwr += info["power_consumed_j"]
            total_data += (info["data_rate_bps"] * 10 / 8) / 1024 # KB sent in 10s
            steps += 1
            done = term or trunc
            
        data_points.append({
            "label": label,
            "power": total_pwr * 1000, # mJ
            "throughput": total_data / 1024 # MB
        })
        
    df_res = pd.DataFrame(data_points)
    
    plt.figure(figsize=(8, 6))
    for i, row in df_res.iterrows():
        color = 'blue' if 'RL' in row['label'] else 'gray'
        marker = 'D' if 'RL' in row['label'] else 'o'
        plt.scatter(row['throughput'], row['power'], s=150, color=color, label=row['label'], marker=marker, edgecolors='black')
        plt.text(row['throughput'] + 0.05, row['power'] + 0.5, row['label'], fontsize=10)
        
    plt.xlabel("Total Data Throughput (MB/Pass)")
    plt.ylabel("Total Energy Consumed (mJ/Pass)")
    plt.title("B. Energy-Throughput Pareto Efficiency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "Paper_PlotB_Pareto.png"), dpi=300)
    print("Saved Plot B.")

def generate_plot_c_dynamic():
    print("Generating Plot C: Dynamic Adaptivity...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = LEOCommEnv(device=device)
    model = PPO.load(MODEL_PATH, device=device)
    
    obs, _ = env.reset()
    done = False
    
    history = {"time": [], "doppler": [], "action": [], "dist": []}
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        
        # Calculate Doppler Rate for visualization (Physics inform)
        dist_km = info["dist_km"]
        # Approximate Doppler Rate shift
        history["time"].append(env.curr_step * 10 / 60) # mins
        history["action"].append(action[0])
        history["dist"].append(dist_km)
        
        done = term or trunc
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Invert Distance for Zenith focus (Zenith = min distance)
    ax1.plot(history["time"], history["dist"], color='red', linewidth=2, label="Range (km)")
    ax1.set_ylabel("Range (km)")
    ax1.legend(loc='upper right')
    ax1.set_title("C. Physics-Aware RL Decision Dynamics")
    
    # Action
    ax2.plot(history["time"], history["action"], color='blue', linewidth=2, label="Guard Band Ratio (RL Action)")
    ax2.set_ylabel("Guard Band\nRatio")
    ax2.set_xlabel("Pass Time (minutes)")
    ax2.set_ylim(0, 0.6)
    ax2.grid(True)
    ax2.legend(loc='upper right')
    
    # Annotations
    if len(history["time"]) > 10:
        idx = 10
        ax2.annotate('High Precision Area\nShrinking margins', xy=(history["time"][idx], history["action"][idx]), 
                     xytext=(history["time"][idx] + 2, 0.4), arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "Paper_PlotC_DynamicAdaptivity.png"), dpi=300)
    print("Saved Plot C.")

if __name__ == "__main__":
    generate_plot_a_learning_curve()
    generate_plot_b_pareto()
    generate_plot_c_dynamic()
