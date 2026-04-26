"""Tests for PINN core module."""
import pytest, torch
from models.pinn_core import TrajectoryPINN, FourierFeatureEmbedding, SirenLayer, count_parameters


def test_siren_layer():
    layer = SirenLayer(10, 20, omega0=30.0, is_first=True)
    x = torch.randn(32, 10)
    out = layer(x)
    assert out.shape == (32, 20)
    assert not torch.isnan(out).any()


def test_fourier_embedding():
    embed = FourierFeatureEmbedding(1, num_features=32)
    t = torch.randn(16, 1)
    out = embed(t)
    assert out.shape == (16, 64)  # sin + cos


def test_trajectory_pinn_forward():
    model = TrajectoryPINN()
    t = torch.randn(32, 1)
    oe = torch.randn(32, 7)
    out = model(t, oe)
    assert out.shape == (32, 6)
    assert not torch.isnan(out).any()


def test_trajectory_pinn_batch_consistency():
    model = TrajectoryPINN()
    t = torch.randn(1, 1)
    oe = torch.randn(1, 7)
    out1 = model(t, oe)
    t2 = t.repeat(4, 1)
    oe2 = oe.repeat(4, 1)
    out2 = model(t2, oe2)
    # First sample should match
    assert torch.allclose(out1, out2[0:1], atol=1e-5)


def test_parameter_count():
    model = TrajectoryPINN()
    n = count_parameters(model)
    assert n > 100_000  # Should be several hundred thousand params


def test_pinn_gradient_flow():
    model = TrajectoryPINN()
    t = torch.randn(8, 1, requires_grad=True)
    oe = torch.randn(8, 7, requires_grad=True)
    out = model(t, oe)
    loss = out.sum()
    loss.backward()
    assert t.grad is not None
    assert oe.grad is not None
    # Check that gradients flow to trainable parameters
    trainable_grads = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for trainable {name}"
            trainable_grads += 1
    assert trainable_grads >= 2  # at least elem_scale and elem_bias


if __name__ == "__main__":
    pytest.main([__file__, "-v"])