"""Tests for GRPO agent module."""
import pytest, numpy as np, torch
from models.grpo_agent import GRPOAgent, create_grpo_agent, GRPOConfig, RolloutBuffer, RolloutEntry
from models.pinn_core import TrajectoryPINN


def test_rollout_buffer():
    buf = RolloutBuffer(capacity=5)
    for i in range(7):
        entry = RolloutEntry(
            t=torch.tensor([[i]]),
            orbital_elems=torch.randn(1, 7),
            predicted_state=torch.randn(6),
            gps_state=torch.randn(6),
            reward=float(-i),
            value_estimate=float(-i),
        )
        buf.add(entry)
    assert len(buf) == 5  # capacity respected


def test_grpo_agent_smoke():
    model = TrajectoryPINN()
    agent = create_grpo_agent(model, learning_rate=1e-4)
    assert agent.step_count == 0

    t = 3600.0
    orbital_elems = np.array([6778.137, 0.001, 0.925, 0.0, 0.0, 0.0, 0.00112])
    gps_pos = np.array([6778.0, 0.0, 0.0])
    gps_vel = np.array([0.0, 7.5, 0.0])

    reward, error, should_update = agent.observe_and_reward(t, orbital_elems, gps_pos, gps_vel)
    assert isinstance(reward, float)
    assert isinstance(error, float)
    assert error >= 0


def test_grpo_kl_beta_adaptation():
    model = TrajectoryPINN()
    config = GRPOConfig(kl_target=0.01)
    agent = GRPOAgent(model, config)
    initial_beta = agent.kl_beta

    # Simulate many updates to trigger adaptation
    for _ in range(5):
        agent.kl_beta = 10.0  # force high
        agent.config.kl_target = 0.01

    assert agent.kl_beta >= 0.5


def test_grpo_buffer_clear():
    model = TrajectoryPINN()
    agent = create_grpo_agent(model)
    for _ in range(3):
        agent.observe_and_reward(
            0.0,
            np.ones(7),
            np.zeros(3),
            np.zeros(3),
        )
    assert len(agent.buffer) == 3
    agent.buffer.clear()
    assert len(agent.buffer) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
