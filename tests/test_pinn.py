"""Tests for PINN core module."""
import pytest, torch
from physics_ml.pinn_core import TrajectoryPINN, FourierFeatureEmbedding, SirenLayer, count_parameters


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


# ─────────────────────────────────────────────────────────────────────────────
#  Uncertainty head tests (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_output_shape():
    """Deterministic mode (output_uncertainty=False) returns (B, 6)."""
    model = TrajectoryPINN(output_uncertainty=False)
    t = torch.randn(32, 1)
    oe = torch.randn(32, 7)
    out = model(t, oe)
    assert out.shape == (32, 6), f"Expected (32, 6), got {out.shape}"
    assert not torch.isnan(out).any()


def test_uncertainty_output_shape():
    """Uncertainty mode (output_uncertainty=True) returns (B, 12)."""
    model = TrajectoryPINN(output_uncertainty=True)
    t = torch.randn(32, 1)
    oe = torch.randn(32, 7)
    out = model(t, oe)
    assert out.shape == (32, 12), f"Expected (32, 12), got {out.shape}"
    mean, log_var = out[:, :6], out[:, 6:]
    assert not torch.isnan(mean).any()
    assert not torch.isnan(log_var).any()


def test_uncertainty_logvar_clamp():
    """log_var is clamped to [-10, 5] (sigma range ~0.007–12.2 km)."""
    model = TrajectoryPINN(output_uncertainty=True)
    t = torch.randn(64, 1)
    oe = torch.randn(64, 7)
    out = model(t, oe)
    log_var = out[:, 6:]
    assert log_var.min() >= -10.0 - 1e-5, f"log_var min {log_var.min()} < -10"
    assert log_var.max() <= 5.0 + 1e-5, f"log_var max {log_var.max()} > 5"


def test_deterministic_checkpoint_loads():
    """A deterministic checkpoint (6-output final_layer) loads into
    output_uncertainty=False model without modification."""
    det = TrajectoryPINN(output_uncertainty=False)
    t = torch.randn(8, 1)
    oe = torch.randn(8, 7)
    _ = det(t, oe)  # initialise state
    sd = det.state_dict()

    # Load into fresh deterministic model — must work
    det2 = TrajectoryPINN(output_uncertainty=False)
    det2.load_state_dict(sd)  # should not raise
    out2 = det2(t, oe)
    assert out2.shape == (8, 6)


def test_deterministic_checkpoint_loads_into_uncertainty_model():
    """A deterministic 6-output checkpoint can load into an
    output_uncertainty=True model via _load_compatible.
    The mean head gets the deterministic weights; log_var is initialised fresh."""
    # Create and "train" a deterministic model (simulate by running forward)
    det = TrajectoryPINN(output_uncertainty=False)
    t = torch.randn(8, 1)
    oe = torch.randn(8, 7)
    with torch.no_grad():
        det(t, oe)
    sd_det = det.state_dict()

    # Load into an uncertainty model
    unc = TrajectoryPINN(output_uncertainty=True)
    # Verify it has mean_layer and log_var_layer (not final_layer)
    assert hasattr(unc, "mean_layer")
    assert hasattr(unc, "log_var_layer")
    assert not hasattr(unc, "final_layer")

    unc._load_compatible(sd_det)

    # Mean should now match original deterministic output
    with torch.no_grad():
        det_out = det(t, oe)
        unc_out = unc(t, oe)
    det_mean = det_out  # (8, 6)
    unc_mean = unc_out[:, :6]  # (8, 6)
    assert torch.allclose(det_mean, unc_mean, atol=1e-5), \
        "Mean after loading deterministic checkpoint should match original"
    # Log-var should be fresh (not loaded from deterministic checkpoint)
    log_var = unc_out[:, 6:]
    assert not torch.isnan(log_var).all(), "log_var should be initialised, not NaN"


def test_uncertainty_forward_backward():
    """1-batch forward + backward pass on uncertainty model succeeds."""
    model = TrajectoryPINN(output_uncertainty=True, hidden_dim=128, num_layers=4)
    t = torch.randn(4, 1, requires_grad=True)
    oe = torch.randn(4, 7, requires_grad=True)
    target = torch.randn(4, 6)

    out = model(t, oe)  # (4, 12)
    assert out.shape == (4, 12)
    mean, log_var = out[:, :6], out[:, 6:]

    # Gaussian NLL backward
    from physics_ml.losses import gaussian_nll_loss
    loss = gaussian_nll_loss(mean, log_var, target)
    assert not torch.isnan(loss), "Loss should not be NaN"
    loss.backward()

    # Check gradients exist on mean and log_var layers
    assert model.mean_layer.weight.grad is not None
    assert model.log_var_layer.weight.grad is not None
    assert t.grad is not None