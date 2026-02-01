import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import os
import torch
import numpy as np
import pandas as pd
from src.rl.env import LEOCommEnv

# Paths
WEIGHTS_DIR = os.path.join("models", "rl")
LOGS_DIR = "logs"
MODEL_PATH = os.path.join("weights", "f7_hybrid_transfer.pth")
RL_MODEL_SAVE = os.path.join(WEIGHTS_DIR, "rl_policy_mac")

class SimpleLogCallback(BaseCallback):
    """
    Saves rewards to a CSV for Plot A.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.log_file = os.path.join(LOGS_DIR, "rl_training_history.csv")
        os.makedirs(LOGS_DIR, exist_ok=True)

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.rewards.append({
                        "step": self.num_timesteps,
                        "reward": info['episode']['r'],
                        "length": info['episode']['l']
                    })
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.rewards)
        df.to_csv(self.log_file, index=False)
        print(f"Training history saved to {self.log_file}")

def train_rl_agent(total_timesteps=50000):
    print("--- Phase I: RL Agent Training (Autonomous MAC) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize Environment
    # Wrap in Monitor to get episode info for the callback
    from stable_baselines3.common.monitor import Monitor
    env = LEOCommEnv(hybrid_weights_path=MODEL_PATH, device=device)
    env = Monitor(env)
    
    # 2. Define Model (PPO)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device=device
    )
    
    # 3. Training
    callback = SimpleLogCallback()
    print(f"Training PPO agent for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
    
    # 4. Save Model
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model.save(RL_MODEL_SAVE)
    print(f"RL Policy saved to {RL_MODEL_SAVE}.zip")

if __name__ == "__main__":
    train_rl_agent()
