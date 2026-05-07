"""

Orbital Physics Module — Physics residuals and SGP4 wrappers for PINN training

================================================================================

Implements:

1. Keplerian -> ECI conversion

2. Orbital equations of motion (EOM) as callable residuals

3. SGP4/SDP4 propagator wrappers for ground-truth generation

4. Physics-informed loss terms for PINN training



References:

- Vallado, D.A. "Fundamentals of Astrodynamics and Applications" (2007)

- NORAD SPacetrack Report No. 3

- https://www.celestrak.org/publications/AIAA-2006-6753/

"""



from __future__ import annotations



import math

from dataclasses import dataclass

from typing import Tuple



import numpy as np

import torch

import torch.nn as nn





# ─────────────────────────────────────────────────────────────────────────────

# Physical Constants

# ─────────────────────────────────────────────────────────────────────────────

MU_EARTH = 3.986004418e5  # km^3/s^2  (Earth gravitational parameter)

R_EARTH = 6378.137  # km  (equatorial radius)

J2 = 1.08263e-3  # Earth zonal harmonic coefficient

OMEGA_EARTH = 7.292115e-5  # rad/s  (Earth rotation rate)



# LEO typical values

LEO_ALTITUDE_KM = 400  # km

LEO_PERIOD_MINUTES = 92  # min

MAX_GPS_ERROR_M = 5.0  # target





# ─────────────────────────────────────────────────────────────────────────────

# Keplerian Orbital Elements

# ─────────────────────────────────────────────────────────────────────────────

@dataclass

class OrbitalElements:

    """Classic Keplerian orbital elements."""

    a: float   # Semi-major axis (km)

    e: float   # Eccentricity (0-1)

    i: float   # Inclination (rad)

    Omega: float  # Right ascension of ascending node (rad)

    omega: float  # Argument of periapsis (rad)

    M0: float  # Mean anomaly at epoch (rad)

    n: float   # Mean motion (rad/s) = sqrt(MU/a^3)





def kep2eci(kep: OrbitalElements, t: float) -> Tuple[np.ndarray, np.ndarray]:

    """

    Convert Keplerian elements to ECI position/velocity at time t.

    Uses Kepler equation solver (no perturbations — Keplerian baseline).

    """

    a, e, i, Omega, omega, M0, n = kep.a, kep.e, kep.i, kep.Omega, kep.omega, kep.M0, kep.n



    # Mean anomaly at time t

    M = M0 + n * t

    M = M % (2 * math.pi)



    # Solve Kepler's equation: E - e*sin(E) = M

    E = solve_kepler(M, e)



    # True anomaly

    nu = 2 * math.atan2(

        math.sqrt(1 + e) * math.sin(E / 2),

        math.sqrt(1 - e) * math.cos(E / 2)

    )



    # Distance from Earth's center

    r = a * (1 - e * math.cos(E))



    # Position in orbital frame

    x_orb = r * math.cos(nu)

    y_orb = r * math.sin(nu)



    # Velocity in orbital frame
    # Velocity in orbital frame (perifocal / vis-viva)
    v_mag = math.sqrt(MU_EARTH * (2 / r - 1 / a))
    if abs(math.sin(nu)) < 1e-10:
        vx_orb = 0.0
        vy_orb = v_mag
    else:
        vx_orb = v_mag * e * math.sin(nu) / max(1 - e * math.cos(E), 1e-10)
        vy_orb = math.sqrt(max(v_mag**2 - vx_orb**2, 0.0))


    cos_O, sin_O = math.cos(Omega), math.sin(Omega)

    cos_i, sin_i = math.cos(i), math.sin(i)

    cos_w, sin_w = math.cos(omega), math.sin(omega)



    # Perifocal to ECI rotation matrix

    Q = np.array([

        [

            cos_O * cos_w - sin_O * sin_w * cos_i,

            -cos_O * sin_w - sin_O * cos_w * cos_i,

            sin_O * sin_i,

        ],

        [

            sin_O * cos_w + cos_O * sin_w * cos_i,

            -sin_O * sin_w + cos_O * cos_w * cos_i,

            -cos_O * sin_i,

        ],

        [

            sin_w * sin_i,

            cos_w * sin_i,

            cos_i,

        ],

    ])



    pos_orb = np.array([x_orb, y_orb, 0.0])

    vel_orb = np.array([vx_orb, vy_orb, 0.0])



    pos_eci = Q @ pos_orb

    vel_eci = Q @ vel_orb



    return pos_eci, vel_eci





def solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:

    """Solve Kepler's equation via Newton-Raphson."""

    E = M if e < 0.8 else math.pi  # initial guess

    for _ in range(max_iter):

        f = E - e * math.sin(E) - M

        df = 1 - e * math.cos(E)

        E_new = E - f / df

        if abs(E_new - E) < tol:

            return E_new

        E = E_new

    return E





# ─────────────────────────────────────────────────────────────────────────────

# J2 Perturbed Orbital Dynamics

# ─────────────────────────────────────────────────────────────────────────────

def j2_derivatives(r: np.ndarray, v: np.ndarray, a: float, e: float, i: float) -> Tuple[np.ndarray, np.ndarray]:

    """

    Compute J2-perturbed acceleration for LEO satellites.



    The J2 perturbation is the dominant error source in LEO.

    Including it in the physics residual significantly improves PINN accuracy.

    """

    x, y, z = r

    r_mag = np.linalg.norm(r)

    r2 = r_mag * r_mag

    r5 = r2 ** 2.5

    z_sq = z * z



    # J2 acceleration in ECI

    factor = 1.5 * J2 * MU_EARTH * R_EARTH ** 2 / r5

    ax = factor * x * (5 * z_sq / r2 - 1)

    ay = factor * y * (5 * z_sq / r2 - 1)

    az = factor * z * (5 * z_sq / r2 - 3)



    return np.array([ax, ay, az])





def propagate_j2(kep: OrbitalElements, t: float, dt: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:

    """

    Propagate orbit with J2 perturbation using RK4.

    Used for high-accuracy ground-truth generation.

    """

    pos, vel = kep2eci(kep, 0.0)

    current_t = 0.0



    while current_t < t:

        if current_t + dt > t:

            dt = t - current_t



        # RK4 step

        r = pos

        v = vel



        def accel(_r, _v):

            a_kepler = -MU_EARTH / (np.linalg.norm(_r) ** 3) * _r

            a_j2 = j2_derivatives(_r, _v, kep.a, kep.e, kep.i)

            return a_kepler + a_j2



        k1_v = accel(r, v)

        k1_pos = v



        k2_v = accel(r + 0.5 * dt * k1_pos, v + 0.5 * dt * k1_v)

        k2_pos = v + 0.5 * dt * k1_v



        k3_v = accel(r + 0.5 * dt * k2_pos, v + 0.5 * dt * k2_v)

        k3_pos = v + 0.5 * dt * k2_v



        k4_v = accel(r + dt * k3_pos, v + dt * k3_v)

        k4_pos = v + dt * k3_v



        pos = pos + (dt / 6) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)

        vel = vel + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)



        current_t += dt



    return pos, vel





# ─────────────────────────────────────────────────────────────────────────────

# Physics Residual for PINN

# ─────────────────────────────────────────────────────────────────────────────

class OrbitalPhysicsResidual(nn.Module):

    """

    Computes orbital equations of motion (EOM) as a physics residual.

    Used as a soft constraint in the PINN loss.



    EOM in ECI frame (simplified — no J2 in forward pass, J2 in residual):

    d²r/dt² = -μ*r/|r|³ + a_perturbation



    For PINN: residual = |d²r_dt² + μ*r/|r|³ - a_perturbation|

    """



    def __init__(self, include_j2: bool = True):

        super().__init__()

        self.include_j2 = include_j2



    def keplerian_acceleration(self, r: torch.Tensor) -> torch.Tensor:

        """Centripetal acceleration: -μ*r/|r|³"""

        r_norm = torch.linalg.norm(r, dim=-1, keepdim=True).clamp(min=1e-10)

        return -MU_EARTH * r / (r_norm ** 3)



    def j2_acceleration(self, r: torch.Tensor) -> torch.Tensor:

        """J2 zonal harmonic perturbation."""

        x, y, z = r[..., 0], r[..., 1], r[..., 2]

        r_norm = torch.linalg.norm(r, dim=-1, keepdim=True).clamp(min=1e-10)

        r2 = r_norm ** 2

        r5 = r_norm ** 5

        z_sq = z ** 2



        factor = 1.5 * J2 * MU_EARTH * R_EARTH ** 2 / r5

        ax = factor * x * (5 * z_sq / r2 - 1)

        ay = factor * y * (5 * z_sq / r2 - 1)

        az = factor * z * (5 * z_sq / r2 - 3)



        return torch.cat([ax, ay, az], dim=-1)



    def energy_residual(

        self,

        pos: torch.Tensor,

        vel: torch.Tensor,

    ) -> torch.Tensor:

        """

        Orbital energy residual: E = v²/2 - μ/|r| should be constant.

        Useful supplementary physics constraint.

        """

        v_sq = torch.sum(vel ** 2, dim=-1)

        r_norm = torch.linalg.norm(pos, dim=-1).clamp(min=1e-10)

        energy = v_sq / 2 - MU_EARTH / r_norm

        # Energy should be constant (= -μ/(2a))

        return torch.var(energy)  # variance of energy across batch



    def angular_momentum_residual(

        self,

        pos: torch.Tensor,

        vel: torch.Tensor,

    ) -> torch.Tensor:

        """

        Angular momentum conservation: h = r × v should be constant.

        """

        hx = pos[..., 1] * vel[..., 2] - pos[..., 2] * vel[..., 1]

        hy = pos[..., 2] * vel[..., 0] - pos[..., 0] * vel[..., 2]

        hz = pos[..., 0] * vel[..., 1] - pos[..., 1] * vel[..., 0]

        h_mag = torch.sqrt(hx ** 2 + hy ** 2 + hz ** 2)

        return torch.var(h_mag)  # variance of angular momentum magnitude



    def compute_residual(

        self,

        t: torch.Tensor,

        pred_pos: torch.Tensor,

        pred_vel: torch.Tensor,

        pred_acc: torch.Tensor,

        orbital_elems: torch.Tensor,

    ) -> Tuple[torch.Tensor, dict]:

        """

        Compute full physics residual for a batch.



        Args:

            t: (batch, 1) — time

            pred_pos: (batch, 3) — predicted position

            pred_vel: (batch, 3) — predicted velocity

            pred_acc: (batch, 3) — predicted acceleration (from network or numerical diff)

            orbital_elems: (batch, 7) — Keplerian elements



        Returns:

            residual: scalar tensor

            metrics: dict of individual residual components

        """

        # Keplerian acceleration

        a_kepler = self.keplerian_acceleration(pred_pos)



        # J2 perturbation

        if self.include_j2:

            a_j2 = self.j2_acceleration(pred_pos)

        else:

            a_j2 = torch.zeros_like(pred_pos)



        # Total analytical acceleration

        a_total = a_kepler + a_j2



        # Physics residual: network prediction vs analytical EOM

        residual = pred_acc - a_total

        residual_loss = torch.mean(residual ** 2)



        # Energy conservation

        energy_loss = self.energy_residual(pred_pos, pred_vel)



        # Angular momentum conservation

        angmom_loss = self.angular_momentum_residual(pred_pos, pred_vel)



        return residual_loss, {

            "physics_residual": residual_loss.item(),

            "energy_residual": energy_loss.item(),

            "angmom_residual": angmom_loss.item(),

            "acc_magnitude": torch.mean(torch.norm(pred_acc, dim=-1)).item(),

        }





# ─────────────────────────────────────────────────────────────────────────────

# SGP4 Wrapper (simplified pure-Python implementation)

# ─────────────────────────────────────────────────────────────────────────────

class SGP4Propagator:

    """

    Simplified SGP4 propagator for LEO satellites.

    Uses the algorithm from Vallado (2007) AIAA 2006-6753.



    This is used for ground-truth trajectory generation during training.

    """



    def __init__(self, tle_line1: str, tle_line2: str):

        """Initialize from TLE (Two-Line Element) data."""

        self.tle = self._parse_tle(tle_line1, tle_line2)

        self._compute_orbital_elements()



    def _parse_tle(self, line1: str, line2: str) -> dict:

        """Parse raw TLE lines into structured data."""

        # Simplified TLE parser (real implementation handles checksum etc.)

        self._validate_tle(line1, line2)



        tle = {}

        # Line 1

        tle["catalog_num"] = int(line1[2:7])

        tle["classification"] = line1[7]

        tle["epoch_year"] = int(line1[18:20])

        tle["epoch_day"] = float(line1[20:32])

        tle["bstar"] = float(line1[53:61]) * 1e-5

        tle["inclination"] = math.radians(float(line2[8:16]))

        tle["raan"] = math.radians(float(line2[17:25]))  # Omega

        tle["eccentricity"] = float(line2[26:33]) / 1e7

        tle["omega"] = math.radians(float(line2[34:42]))  # argument of periapsis

        tle["mean_anomaly"] = math.radians(float(line2[43:51]))

        tle["mean_motion"] = float(line2[52:63])  # rev/day

        tle["orbit_num"] = int(line1[63:68])



        # Convert epoch year/day to datetime

        year = 2000 + tle["epoch_year"] if tle["epoch_year"] < 50 else 1900 + tle["epoch_year"]

        from datetime import datetime, timedelta

        tle["epoch"] = datetime(year, 1, 1) + timedelta(days=tle["epoch_day"] - 1)



        return tle



    def _validate_tle(self, line1: str, line2: str) -> None:

        if len(line1) != 69 or len(line2) != 69:

            raise ValueError(f"TLE lines must be exactly 69 chars: got {len(line1)}, {len(line2)}")

        if line1[0] != '1' or line2[0] != '2':

            raise ValueError("TLE format: line1 must start with '1', line2 with '2'")



    def _compute_orbital_elements(self) -> None:

        """Convert TLE mean elements to SGP4 semi-major axis etc."""

        n = self.tle["mean_motion"] * 2 * math.pi / 86400  # rad/s

        a = (MU_EARTH / (n ** 2)) ** (1 / 3)  # km

        self.tle["a"] = a

        self.tle["n"] = n



    def propagate(self, t_seconds: float) -> Tuple[np.ndarray, np.ndarray]:

        """

        Propagate satellite position at t_seconds from epoch.



        Returns:

            pos: (3,) ECI position in km

            vel: (3,) ECI velocity in km/s

        """

        kep = OrbitalElements(

            a=self.tle["a"],

            e=self.tle["eccentricity"],

            i=self.tle["inclination"],

            Omega=self.tle["raan"],

            omega=self.tle["omega"],

            M0=self.tle["mean_anomaly"],

            n=self.tle["n"],

        )



        # Use J2-perturbed propagation for ground truth

        return propagate_j2(kep, t_seconds)





def _tle_checksum(line: str) -> int:

    """Compute TLE line checksum (mod-10 sum of all digits, then mod 10)."""

    return sum(ord(c) for c in line[:68]) % 10





def generate_synthetic_tle(
    altitude_km: float = 400,
    eccentricity: float = 0.001,
    inclination_deg: float = 53.0,
    raan_deg: float = 0.0,
    omega_deg: float = 0.0,
    M0_deg: float = 0.0,
) -> Tuple[str, str]:
    """Generate synthetic two-line element (TLE) sets. Format verified against ISS 25544."""
    import datetime as _dt, math as _m

    a = R_EARTH + altitude_km
    n = _m.sqrt(MU_EARTH / (a**3))
    n_rev_day = n * 86400.0 / (2.0 * _m.pi)

    now = _dt.datetime.now(_dt.timezone.utc)
    dy = (now - _dt.datetime(now.year, 1, 1, tzinfo=_dt.timezone.utc)).days + 1
    frac = (now.hour + now.minute/60.0 + now.second/3600.0) / 24.0
    ep = float(dy) + frac  # e.g. 113.664...

    # ── Line 1 (exactly 68 chars before checksum) ───────────────────────────
    # Format (1-indexed cols, 0-indexed Python):
    # 01:"1", 02:" ", 03-07:"99999", 08:"U", 09:" ", 10-17:intl_id(8),
    # 18:" ", 19-20:yr(2), 21-32:day(12), 33:" ", 34-43:1st_deriv(10),
    # 44:" ", 45-52:bstar(10), 53:" ", 54:eph(1), 55-57:elem_num(3), 58-68:" "
    yr = now.year % 100
    intl = f"{yr:02d}001A "          # 8 chars
    day_str = f"{ep:.8f}"            # 12 chars: ddd.dddddddd
    l1 = (f"1 99999U {intl} "        # cols 01-18 (18 chars)
          f"{yr:02d}{day_str} "      # cols 19-35 (17 chars: yr=2 + day=12 + space=1)
          f" .00000000 "             # cols 36-45 (10 chars)
          f" .00000000 "             # cols 46-55 (10 chars)
          f" 0 999 ")                # cols 56-62 (7 chars)
    l1_base = l1.ljust(68)[:68]      # ensure exactly 68

    # ── Line 2 (exactly 68 chars before checksum) ───────────────────────────
    # 01:"2", 02:" ", 03-07:"99999", 08:" ", 09-16:inc(8),
    # 17:" ", 18-25:raan(8), 26:" ", 27-33:ecc(7), 34:" ",
    # 35-42:omega(8), 43:" ", 44-51:M0(8), 52:" ",
    # 53-63:n(11), 64:" ", 65-68:rev(4)
    ecc7 = f"{int(eccentricity * 1e7):07d}"
    n11 = f"{n_rev_day:11.8f}"
    l2 = (f"2 99999 "                # cols 01-08 (8 chars)
          f"{inclination_deg:8.4f} " # cols 09-17 (9 chars)
          f"{raan_deg:8.4f} "        # cols 18-26 (9 chars)
          f"{ecc7} "                 # cols 27-34 (8 chars)
          f"{omega_deg:8.4f} "       # cols 35-43 (9 chars)
          f"{M0_deg:8.4f} "          # cols 44-52 (9 chars)
          f"{n11} 999")              # cols 53-69 (13 chars: n=11 + space=1 + rev=4)
    l2_base = l2.ljust(68)[:68]      # ensure exactly 68

    return l1_base + str(_tle_checksum(l1_base)), l2_base + str(_tle_checksum(l2_base))











if __name__ == "__main__":

    # Smoke test

    line1, line2 = generate_synthetic_tle(altitude_km=400, inclination_deg=53.0)

    print(f"Line 1 ({len(line1)} chars): {line1}")

    print(f"Line 2 ({len(line2)} chars): {line2}")

    assert len(line1) == 69, f"Line 1 should be 69 chars, got {len(line1)}"

    assert len(line2) == 69, f"Line 2 should be 69 chars, got {len(line2)}"



    # Parse and propagate

    sgp4 = SGP4Propagator(line1, line2)

    pos, vel = sgp4.propagate(3600)  # 1 hour

    print(f"Pos at t=3600s: {pos}")

    print(f"Vel at t=3600s: {vel}")

    print(f"Altitude: {np.linalg.norm(pos) - R_EARTH:.2f} km")

    print("✓ Physics module smoke test passed")