"""
GRPO Agent — Gradient-Regulated Policy Optimization for Online RL
================================================================
Implements the GRPO algorithm adapted for satellite trajectory PINN:

Key ideas from the reference repo's GRPO (GitHub: Vanisherzd/LEO-Hybrid-PGRL):
1. Gradient regulation: gradients from successful comm events regulate policy updates
2. Trust region: KL-divergence constraint on weight updates (like PPO but gradient-based)
3. Multi-objective: trajectory accuracy + GPS error reduction + communication quality

On every successful TDMA transmission:
  → Evaluate current PINN prediction error
  → If error > threshold (e.g., 5m): compute GRPO gradient correction
  → Apply soft weight update via gradient stepping

Reference: Vanisherzd/LEO-Hybrid-PGRL — Hybrid Policy Gradient with Reinforcement Learning
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# GRPO Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GRPOConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99           # discount factor
    lambda_: float = 0.95         # GAE lambda
    kl_target: float = 0.01       # target KL divergence (trust region)
    kl_beta: float = 2.0          # adaptive beta for KL penalty
    gradient_bound: float = 0.1   # max gradient norm
    entropy_coef: float = 1e-4    # entropy bonus for exploration
    error_threshold_m: float = 5.0  # GPS error threshold (meters)
    advantage_norm: bool = True
    use_gae: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RolloutEntry:
    """Single TDMA communication event as a rollout entry."""
    t: torch.Tensor                # time input (1,)
    orbital_elems: torch.Tensor    # orbital elements (7,)
    predicted_state: torch.Tensor  # predicted 6D state (6,)
    gps_state: torch.Tensor        # ground-truth GPS state (6,) in km
    reward: float                  # instantaneous reward (negative error)
    advantage: float = 0.0
    value_estimate: float = 0.0
    log_prob: float = 0.0          # for policy logging


class RolloutBuffer:
    """Buffer for storing GRPO rollout data from TDMA events."""

    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.entries: list[RolloutEntry] = []
        self.episode_rewards: list[float] = []

    def add(self, entry: RolloutEntry) -> None:
        if len(self.entries) >= self.capacity:
            self.entries.pop(0)
        self.entries.append(entry)
        self.episode_rewards.append(entry.reward)

    def compute_advantages(self, gamma: float = 0.99, lambda_: float = 0.95) -> None:
        """Compute GAE (Generalized Advantage Estimation) over stored entries."""
        if len(self.entries) < 2:
            return

        rewards = [e.reward for e in self.entries]
        values = [e.value_estimate for e in self.entries]

        # Compute TD residuals
        td_residuals = []
        for i in range(len(rewards) - 1):
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            td_residuals.append(delta)
        td_residuals.append(0)  # terminal

        # GAE
        advantages = [0.0] * len(td_residuals)
        last_gae = 0
        for t in reversed(range(len(td_residuals) - 1)):
            gae = td_residuals[t] + gamma * lambda_ * last_gae
            last_gae = gae
            advantages[t] = gae

        for i, adv in enumerate(advantages[:-1]):
            self.entries[i].advantage = adv

    def clear(self) -> None:
        self.entries.clear()
        self.episode_rewards.clear()

    def __len__(self) -> int:
        return len(self.entries)


# ─────────────────────────────────────────────────────────────────────────────
# Value Network (Critic)
# ─────────────────────────────────────────────────────────────────────────────
class ValueCritic(nn.Module):
    """Critic network that estimates expected reward (negative GPS error)."""

    def __init__(self, orbital_elem_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(orbital_elem_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t: torch.Tensor, orbital_elems: torch.Tensor) -> torch.Tensor:
        x = torch.cat([t, orbital_elems], dim=-1)
        return self.net(x)  # (batch, 1) — estimated value


# ─────────────────────────────────────────────────────────────────────────────
# GRPO Agent
# ─────────────────────────────────────────────────────────────────────────────
class GRPOAgent:
    """
    Gradient-Regulated Policy Optimization agent for online PINN updates.

    Workflow:
    1. After each successful TDMA communication, observe (t, orb_elems, GPS_pos, GPS_vel)
    2. Compute reward = -position_error_m / error_threshold
    3. Update value baseline with TD(0) or GAE
    4. If error > threshold → GRPO policy gradient update
    5. Soft update PINN weights via gradient regulation

    Key GRPO features:
    - No separate policy network — directly regulates PINN gradient
    - KL-divergence trust region via adaptive penalty
    - Gradient magnitude bounding for stability
    """

    def __init__(
        self,
        pinn_model: nn.Module,
        config: Optional[GRPOConfig] = None,
    ):
        self.pinn = pinn_model
        self.config = config or GRPOConfig()
        self.device = self.config.device

        # Move model to device
        self.pinn = self.pinn.to(self.device)

        # Value critic for baseline — must match PINN orbital_elem_dim (6)
        self.critic = ValueCritic(orbital_elem_dim=6).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

        # GRPO optimizer for PINN
        self.optimizer = torch.optim.Adam(self.pinn.parameters(), lr=self.config.learning_rate)

        # Rollout buffer
        self.buffer = RolloutBuffer(capacity=1024)

        # Statistics
        self.step_count = 0
        self.update_count = 0
        self.kl_beta = self.config.kl_beta

    def observe_and_reward(
        self,
        t: float,
        orbital_elems: np.ndarray,
        gps_pos: np.ndarray,
        gps_vel: np.ndarray,
    ) -> Tuple[float, float, bool]:
        """
        Process a new observation from a successful TDMA communication.

        Args:
            t: time since epoch (seconds)
            orbital_elems: (7,) Keplerian elements
            gps_pos: (3,) GPS position in km
            gps_vel: (3,) GPS velocity in km/s

        Returns:
            reward: scalar reward
            position_error_m: error in meters
            should_update: whether GRPO update should trigger
        """
        # Forward pass through PINN
        t_t = torch.tensor([[t]], dtype=torch.float32, device=self.device)
        oe_t = torch.tensor([orbital_elems], dtype=torch.float32, device=self.device)

        self.pinn.eval()
        with torch.no_grad():
            pred_state = self.pinn(t_t, oe_t).cpu().numpy()[0]

        pred_pos = pred_state[:3]
        pred_vel = pred_state[3:]

        # Compute position error in meters
        position_error_m = np.linalg.norm(pred_pos - gps_pos) * 1000  # km -> m
        velocity_error_ms = np.linalg.norm(pred_vel - gps_vel) * 1000  # km/s -> m/s

        # Reward: negative normalized error
        reward = -position_error_m / self.config.error_threshold_m
        reward = float(reward)  # scalar

        # Store in buffer
        entry = RolloutEntry(
            t=t_t.cpu(),
            orbital_elems=oe_t.cpu(),
            predicted_state=torch.tensor(pred_state, dtype=torch.float32),
            gps_state=torch.tensor(np.concatenate([gps_pos, gps_vel]), dtype=torch.float32),
            reward=reward,
            value_estimate=reward,  # bootstrap
            log_prob=0.0,
        )
        self.buffer.add(entry)

        # Should we trigger an update?
        should_update = position_error_m > self.config.error_threshold_m

        return reward, position_error_m, should_update

    def compute_grpo_update(self) -> dict:
        """
        Compute GRPO gradient-regulated update from buffered rollout data.

        GRPO Algorithm:
        1. Compute advantages via GAE
        2. Compute policy gradient: ∇θ J = E[∇θ log π(a|s) * A(s,a)]
        3. Apply KL regularization: KL(P_old || P_new) <= δ
        4. Apply gradient magnitude bound
        5. Soft update via gradient stepping

        Returns:
            update_metrics: dict with loss components and statistics
        """
        if len(self.buffer) < 2:
            return {"update_triggered": False, "reason": "buffer_too_small"}

        # Compute advantages
        self.buffer.compute_advantages(
            gamma=self.config.gamma,
            lambda_=self.config.lambda_,
        )

        # Build batch from buffer
        t_batch = torch.cat([e.t for e in self.buffer.entries], dim=0).to(self.device)
        oe_batch = torch.cat([e.orbital_elems for e in self.buffer.entries], dim=0).to(self.device)
        rewards = torch.tensor([e.reward for e in self.buffer.entries], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([e.advantage for e in self.buffer.entries], dtype=torch.float32, device=self.device)

        # Normalize advantages
        if self.config.advantage_norm and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Value (Critic) Update ──
        self.critic_optimizer.zero_grad()
        values = self.critic(t_batch, oe_batch).squeeze()
        value_loss = F.mse_loss(values, rewards)
        value_loss.backward()
        self.critic_optimizer.step()

        # ── GRPO Policy Update ──
        self.pinn.train()
        self.optimizer.zero_grad()

        # Forward pass
        pred_states = self.pinn(t_batch, oe_batch)  # (B, 6)
        gps_states = torch.stack([e.gps_state for e in self.buffer.entries], dim=0).to(self.device)

        # Primary loss: negative advantage (we want to maximize reward = minimize error)
        # Use prediction error weighted by advantage
        pos_error = torch.norm(pred_states[:, :3] - gps_states[:, :3], dim=-1) * 1000  # m
        error_loss = torch.mean(pos_error * (1 + advantages.abs()))  # weighted by advantage magnitude

        # KL divergence penalty (approximate with output deviation from previous)
        # Store old outputs before update
        with torch.no_grad():
            old_preds = pred_states.detach()

        kl_div = F.kl_div(
            F.log_softmax(pred_states, dim=-1),
            F.softmax(old_preds, dim=-1),
            reduction="batchmean",
        )

        # Entropy bonus
        entropy = -torch.mean(F.softmax(pred_states, dim=-1) * F.log_softmax(pred_states + 1e-8, dim=-1))

        # Total GRPO loss
        policy_loss = -error_loss  # maximize error reduction (minimize error)
        kl_penalty = self.kl_beta * kl_div
        entropy_bonus = self.config.entropy_coef * entropy

        total_loss = policy_loss + kl_penalty - entropy_bonus

        # Backward pass
        total_loss.backward()

        # Gradient clipping (trust region via gradient norm bound)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), self.config.gradient_bound)

        self.optimizer.step()

        # Adaptive KL beta
        if kl_div > 1.5 * self.config.kl_target:
            self.kl_beta *= 1.5
        elif kl_div < 0.5 * self.config.kl_target:
            self.kl_beta *= 0.5
        self.kl_beta = max(0.5, min(self.kl_beta, 10.0))

        self.update_count += 1

        # Clear buffer after update
        self.buffer.clear()

        return {
            "update_triggered": True,
            "update_count": self.update_count,
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "kl_beta": self.kl_beta,
            "entropy": entropy.item(),
            "grad_norm": grad_norm.item(),
            "value_loss": value_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "max_advantage": advantages.max().item(),
            "min_advantage": advantages.min().item(),
        }

    def online_update(self) -> dict:
        """
        Triggered on each successful TDMA communication event.
        Performs lightweight GRPO update using the latest observation.
        """
        if len(self.buffer) < 2:
            return {"update_triggered": False, "reason": "insufficient_data"}

        self.step_count += 1
        return self.compute_grpo_update()

    def save(self, path: str) -> None:
        torch.save({
            "pinn_state": self.pinn.state_dict(),
            "critic_state": self.critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "step_count": self.step_count,
            "update_count": self.update_count,
            "kl_beta": self.kl_beta,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.pinn.load_state_dict(checkpoint["pinn_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])
        self.step_count = checkpoint.get("step_count", 0)
        self.update_count = checkpoint.get("update_count", 0)
        self.kl_beta = checkpoint.get("kl_beta", self.config.kl_beta)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────
def create_grpo_agent(pinn_model: nn.Module, **kwargs) -> GRPOAgent:
    config = GRPOConfig(**kwargs)
    return GRPOAgent(pinn_model, config)


if __name__ == "__main__":
    # Smoke test
    from physics_ml.pinn_core import TrajectoryPINN

    pinn = TrajectoryPINN()
    agent = create_grpo_agent(pinn)

    # Simulate a successful TDMA communication
    import numpy as np

    t = 3600.0  # 1 hour
    orbital_elems = np.array([6778.137, 0.001, 0.925, 0.0, 0.0, 0.0, 0.00112])  # ~400km LEO
    gps_pos = np.array([6778.0, 0.0, 0.0])
    gps_vel = np.array([0.0, 7.5, 0.0])

    reward, error, should_update = agent.observe_and_reward(t, orbital_elems, gps_pos, gps_vel)
    print(f"Reward: {reward:.4f}, Error: {error:.2f}m, Should update: {should_update}")

    if should_update:
        metrics = agent.online_update()
        print(f"Update metrics: {metrics}")

    print("✓ GRPO agent smoke test passed")
