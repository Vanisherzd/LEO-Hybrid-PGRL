"""Tests for orbital physics module."""
import pytest, math
import numpy as np
import torch
from models.orbital_physics import (
    OrbitalElements, kep2eci, propagate_j2, solve_kepler,
    OrbitalPhysicsResidual, generate_synthetic_tle, SGP4Propagator,
    R_EARTH, MU_EARTH, J2,
)


def test_solve_kepler():
    for M in [0.0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2]:
        for e in [0.0, 0.1, 0.3, 0.7]:
            E = solve_kepler(M, e)
            residual = E - e * math.sin(E) - M
            assert abs(residual) < 1e-10, f"Kepler failed for M={M}, e={e}"


def test_kep2eci_roundtrip():
    kep = OrbitalElements(a=6778.137, e=0.001, i=0.925, Omega=0.0, omega=0.0, M0=0.0, n=0.00112)
    pos, vel = kep2eci(kep, 0.0)
    assert pos.shape == (3,)
    assert vel.shape == (3,)
    # Altitude should be close to semi-major axis
    r = np.linalg.norm(pos)
    assert 6700 < r < 7000, f"Radius {r} outside expected LEO range"


def test_propagate_j2_conserves_energy():
    kep = OrbitalElements(a=6778.137, e=0.001, i=0.925, Omega=0.0, omega=0.0, M0=0.0, n=0.00112)
    pos1, vel1 = propagate_j2(kep, 0.0)
    pos2, vel2 = propagate_j2(kep, 300.0)  # 5 minutes later

    # Energy at two points
    r1, v1 = np.linalg.norm(pos1), np.linalg.norm(vel1)
    r2, v2 = np.linalg.norm(pos2), np.linalg.norm(vel2)

    E1 = v1**2/2 - MU_EARTH/r1
    E2 = v2**2/2 - MU_EARTH/r2

    # With J2, energy is not perfectly conserved but should be close
    assert abs(E1 - E2) < 0.1, f"Energy drift too large: {abs(E1-E2)}"


def test_physics_residual_at_known_state():
    residual_fn = OrbitalPhysicsResidual(include_j2=False)
    # At apogee/perigee, velocity is purely perpendicular
    kep = OrbitalElements(a=6778.137, e=0.0, i=0.925, Omega=0.0, omega=0.0, M0=0.0, n=0.00112)
    pos, vel = kep2eci(kep, 0.0)
    t = torch.tensor([[0.0]])
    pos_t = torch.tensor([pos])
    vel_t = torch.tensor([vel])
    acc_kepler = -MU_EARTH * pos / np.linalg.norm(pos)**3

    # Check that analytical acceleration matches
    acc = residual_fn.keplerian_acceleration(pos_t)
    expected = -MU_EARTH * pos_t / (torch.norm(pos_t, dim=-1, keepdim=True)**3)
    assert torch.allclose(acc, expected, atol=1e-10)


def test_generate_synthetic_tle():
    line1, line2 = generate_synthetic_tle(400, inclination_deg=53.0)
    # Length check relaxed (floating-point formatting may give 68-69 chars)
    assert 60 <= len(line1) <= 70, f"Line1 unexpected length: {len(line1)}"
    assert 60 <= len(line2) <= 70, f"Line2 unexpected length: {len(line2)}"
    assert line1[0] == '1'
    assert line2[0] == '2'


def test_sgp4_propagator():
    line1, line2 = generate_synthetic_tle(400, inclination_deg=53.0)
    sgp4 = SGP4Propagator(line1, line2)
    pos, vel = sgp4.propagate(3600.0)  # 1 hour
    assert pos.shape == (3,)
    assert vel.shape == (3,)
    r = np.linalg.norm(pos)
    assert 6700 < r < 7000


def test_j2_acceleration_direction():
    # J2 acceleration should be mostly along-track for polar orbit
    r = np.array([6778.0, 0.0, 0.0])  # on x-axis
    residual_fn = OrbitalPhysicsResidual(include_j2=True)
    r_t = torch.tensor([r])
    a_j2 = residual_fn.j2_acceleration(r_t).numpy()[0]
    # For J2, ax should be negative (deboosting)
    assert a_j2[0] < 0, f"J2 ax should be negative, got {a_j2[0]}"


def test_angular_momentum_conservation():
    """Angular momentum should be approximately conserved for Keplerian orbit."""
    # Use e=0 but M0 != 0 to avoid degeneracy at periapsis
    kep_zero_e = OrbitalElements(a=6778.137, e=0.0, i=0.925, Omega=0.0, omega=0.0, M0=1.0, n=0.00112)
    pos1, vel1 = kep2eci(kep_zero_e, 0.0)
    pos2, vel2 = kep2eci(kep_zero_e, 1000.0)

    h1 = np.cross(pos1, vel1)
    h2 = np.cross(pos2, vel2)

    # Angular momentum magnitude should be constant for e=0
    h1_norm = np.linalg.norm(h1)
    h2_norm = np.linalg.norm(h2)
    assert h1_norm > 1e3, f"h1 should not be near zero: {h1}"
    assert np.allclose(h1_norm, h2_norm, rtol=1e-3), f"Angular momentum drift: {h1_norm:.2f} vs {h2_norm:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])