import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import datetime
from src.rl.comm_model import LEOCommModel
from src.models.residual_net import SGP4ErrorCorrector
from src.pinn.utils import Normalizer
from sgp4.api import Satrec, WGS72, jday

class LEOCommEnv(gym.Env):
    """
    Gymnasium Environment for LEO Resource Management.
    The agent learns to optimize Guard Band and Power for an IoT terminal.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, hybrid_weights_path=None, device='cpu'):
        super().__init__()
        self.device = device
        self.comm_model = LEOCommModel(device=device)
        self.normalizer = Normalizer()
        
        # 1. Action Space: [guard_band_ratio] (Continuous 0.05 to 0.5)
        # We can expand this later to [guard_band_ratio, tx_power]
        self.action_space = spaces.Box(low=0.05, high=0.5, shape=(1,), dtype=np.float32)
        
        # 2. Observation Space: [dist_km, range_rate, snr_prev, buffer_status, time_to_los]
        # Normalized observation space
        self.observation_space = spaces.Box(low=-100, high=100, shape=(5,), dtype=np.float32)
        
        # 3. Hybrid Model Loading
        self.hybrid_model = SGP4ErrorCorrector(hidden_dim=512).to(device)
        if hybrid_weights_path and os.path.exists(hybrid_weights_path):
            self.hybrid_model.load_state_dict(torch.load(hybrid_weights_path, map_location=device, weights_only=True))
            self.hybrid_model.eval()
            print(f"Loaded Hybrid Predictor from {hybrid_weights_path}")
        
        # 4. Scenario State
        self.curr_step = 0
        self.max_steps = 600 # 100 minutes at 10s steps
        self.dt = 10.0
        
        # Ground Station (Taiwan GS)
        self.r_gs = np.array([3952.0, 3133.0, 3915.0]) # Approx Taiwan ECI
        self.v_gs = np.array([0, 0, 0]) # Inertial approx
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.curr_step = 0
        self.buffer = 100.0 # Initial data in bits
        
        # Initial State (Formosat-7 Approx)
        # We start with a high-accuracy state near zenith
        self.r_sat_base = np.array([1381.0, -767.0, 7003.0])
        self.v_sat_base = np.array([1.2, 7.5, -0.5])
        
        self.state = self._get_obs()
        return self.state, {}

    def _get_obs(self):
        # 1. Use Hybrid Model to "Predict" Current Displacement Error
        # In a real scenarios, the environment manages the "Ground Truth"
        # and the agent sees the "Corrected SGP4".
        
        # For simplicity in Phase I:
        # Agent sees Distance, Range Rate, and Doppler from the Hybrid Predictor.
        dist_km = np.linalg.norm(self.r_sat_base - self.r_gs)
        range_rate = np.dot(self.r_sat_base - self.r_gs, self.v_sat_base - self.v_gs) / dist_km
        
        # Normalize for RL
        obs = np.array([
            (dist_km - 2000)/1000,
            range_rate/7.0,
            0.0, # SNR prev
            self.buffer / 100.0,
            (self.max_steps - self.curr_step) / self.max_steps
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        guard_band = action[0]
        
        # 1. Predict next state using simplified physics (RK4 logic or pre-computed)
        # Here we just propagate linearly for the demo environment
        self.r_sat_base += self.v_sat_base * self.dt
        # Basic gravity pull to keep it orbiting loosely
        acc = -398600 / (np.linalg.norm(self.r_sat_base)**3) * self.r_sat_base
        self.v_sat_base += acc * self.dt
        
        # 2. Physics Outcome (Link Layer)
        metrics = self.comm_model.calculate_link_metrics(
            self.r_sat_base, self.v_sat_base,
            self.r_gs, self.v_gs,
            tx_power_dbm=20, guard_band_ratio=guard_band
        )
        
        # 3. Handle Data
        success = np.random.random() < metrics["success_prob"]
        data_sent = 0
        if success:
            data_sent = metrics["data_rate_bps"] * self.dt / 8.0 # Bytes
            self.buffer = max(0, self.buffer - data_sent)
            
        # 4. Reward Function
        # Reward throughput, penalize power and failure
        reward = (data_sent / 1000.0) - (metrics["power_consumed_j"] * 10)
        if not success:
            reward -= 0.5 # Penalty for collision/link failure
            
        self.curr_step += 1
        terminated = self.curr_step >= self.max_steps or self.buffer <= 0
        truncated = False
        
        self.state = self._get_obs()
        # Update SNR_prev in obs
        self.state[2] = metrics["snr_db"] / 40.0
        
        info = metrics
        info["success"] = success
        
        return self.state, float(reward), terminated, truncated, info

    def render(self):
        pass

if __name__ == "__main__":
    env = LEOCommEnv()
    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step: {env.curr_step} | Reward: {reward:.4f} | Success: {info['success']}")
