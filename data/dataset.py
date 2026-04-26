"""
TLE Data Module — Two-Line Element parser and dataset generation
"""

from __future__ import annotations

import math, os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# TLE Parser
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TLEData:
    catalog_num: int
    classification: str
    epoch: datetime
    mean_motion: float      # rev/day
    eccentricity: float
    inclination: float      # rad
    raan: float             # rad  (Omega)
    omega: float            # rad  (argument of periapsis)
    mean_anomaly: float     # rad  (M0)
    bstar: float
    orbit_num: int
    semi_major_axis: float  # km
    mean_motion_rad_s: float


def parse_tle(line1: str, line2: str) -> TLEData:
    if len(line1) != 69 or len(line2) != 69:
        raise ValueError("TLE lines must be exactly 69 characters")
    if line1[0] != '1' or line2[0] != '2':
        raise ValueError("Invalid TLE format")

    catalog_num = int(line1[2:7])
    classification = line1[7]
    epoch_year = int(line1[18:20])
    epoch_day = float(line1[20:32])
    bstar = float(line1[53:61]) * 1e-5
    orbit_num = int(line1[63:68])

    inclination = math.radians(float(line2[8:16]))
    raan = math.radians(float(line2[17:25]))
    eccentricity = float(line2[26:33]) / 1e7
    omega = math.radians(float(line2[34:42]))
    mean_anomaly = math.radians(float(line2[43:51]))
    mean_motion = float(line2[52:63])  # rev/day

    # Epoch
    year = 2000 + epoch_year if epoch_year < 50 else 1900 + epoch_year
    epoch = datetime(year, 1, 1) + timedelta(days=epoch_day - 1)

    # Semi-major axis
    n_rad_s = mean_motion * 2 * math.pi / 86400
    a = (3.986004418e5 / (n_rad_s ** 2)) ** (1 / 3)

    return TLEData(
        catalog_num=catalog_num,
        classification=classification,
        epoch=epoch,
        mean_motion=mean_motion,
        eccentricity=eccentricity,
        inclination=inclination,
        raan=raan,
        omega=omega,
        mean_anomaly=mean_anomaly,
        bstar=bstar,
        orbit_num=orbit_num,
        semi_major_axis=a,
        mean_motion_rad_s=n_rad_s,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
class OrbitalDataset(Dataset):
    """
    Dataset for satellite trajectory training.

    Generates (t, orbital_elements) -> (position, velocity) training pairs
    using J2-perturbed propagation as ground truth.

    The dataset creates time sequences around each TLE epoch, covering
    multiple orbital periods for robust learning.
    """

    def __init__(
        self,
        tle_data: TLEData,
        num_samples: int = 10000,
        orbit_seconds: float = 5600,  # ~1 orbit period
        use_j2: bool = True,
        seed: int = 42,
    ):
        self.tle = tle_data
        self.num_samples = num_samples
        self.orbit_seconds = orbit_seconds
        self.use_j2 = use_j2
        self.rng = np.random.default_rng(seed)

        # Pre-generate all samples
        self.times = self.rng.uniform(0, orbit_seconds, num_samples)
        self.times.sort()

        # Orbital elements as array: (a, e, i, Omega, omega, M0, n)
        self.orbital_elements = np.array([
            tle_data.semi_major_axis,
            tle_data.eccentricity,
            tle_data.inclination,
            tle_data.raan,
            tle_data.omega,
            tle_data.mean_anomaly,
            tle_data.mean_motion_rad_s,
        ], dtype=np.float32)

        # Pre-compute ground truth
        self._propagator = None
        self._compute_ground_truth()

    def _compute_ground_truth(self) -> None:
        """Pre-compute ground truth using J2-perturbed propagator."""
        from models.orbital_physics import OrbitalElements, propagate_j2, kep2eci

        if self.use_j2:
            kep = OrbitalElements(
                a=self.orbital_elements[0],
                e=self.orbital_elements[1],
                i=self.orbital_elements[2],
                Omega=self.orbital_elements[3],
                omega=self.orbital_elements[4],
                M0=self.orbital_elements[5],
                n=self.orbital_elements[6],
            )

        positions = []
        velocities = []
        for t in self.times:
            if self.use_j2:
                pos, vel = propagate_j2(kep, t)
            else:
                pos, vel = kep2eci(kep, t)
            positions.append(pos)
            velocities.append(vel)

        self.gt_positions = np.array(positions, dtype=np.float32)
        self.gt_velocities = np.array(velocities, dtype=np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.tensor([[self.times[idx]]], dtype=torch.float32)
        orbital_elems = torch.tensor(self.orbital_elements, dtype=torch.float32).unsqueeze(0)
        position = torch.tensor(self.gt_positions[idx], dtype=torch.float32)
        velocity = torch.tensor(self.gt_velocities[idx], dtype=torch.float32)
        # State = [x, y, z, vx, vy, vz]
        state = torch.cat([position, velocity])
        return t, orbital_elems, state


class NPZOrbitalDataset(Dataset):
    """Load orbital data from .npz files generated by generate_data.py.
    Applies normalization to orbital elements and states for stable training.
    """

    # Normalization constants
    _OE_MEAN = np.array([6778.137, 0.001, 0.925, 0.0, 0.0, 0.0])
    _OE_STD  = np.array([1.0,     0.001,  0.35, 2.0, 2.0, 2.0])
    _STATE_SCALE_POS = 1e4
    _STATE_SCALE_VEL = 10.0
    _T_SCALE = 16661.0  # max time for 1 orbit at 400km

    def __init__(self, npz_dir: str, split: str = "train", train_ratio: float = 0.8):
        self.split = split
        npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
        assert npz_files, f"No .npz files found in {npz_dir}"

        # Split files into train/val
        n_train = int(len(npz_files) * train_ratio)
        if split == "train":
            files = npz_files[:n_train]
        else:
            files = npz_files[n_train:]

        self.samples = []
        for fname in files:
            data = np.load(os.path.join(npz_dir, fname))
            times = data["times"]          # (N,)
            positions = data["positions"]  # (N, 3) km
            velocities = data["velocities"]  # (N, 3) km/s
            orbital_elements = data["orbital_elements"]  # (7,)

            for i in range(len(times)):
                self.samples.append({
                    "t": times[i],
                    "pos": positions[i],
                    "vel": velocities[i],
                    "oe": orbital_elements,
                    "fname": fname,
                })

        print(f"  [{split}] loaded {len(self.samples)} samples from {len(files)} files")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        t_norm = s["t"] / self._T_SCALE
        t = torch.tensor([[t_norm]], dtype=torch.float32)
        oe_raw = s["oe"][:6]
        oe_norm = (oe_raw - self._OE_MEAN) / (self._OE_STD + 1e-8)
        oe = torch.tensor(oe_norm, dtype=torch.float32).unsqueeze(0)
        state = np.concatenate([s["pos"], s["vel"]])
        state_norm = state.copy()
        state_norm[:3] /= self._STATE_SCALE_POS
        state_norm[3:] /= self._STATE_SCALE_VEL
        return t, oe, torch.tensor(state_norm, dtype=torch.float32).view(1, 6)


def create_dataloader(npz_dir: str, split: str, batch_size: int = 256, num_workers: int = 4):
    """Create a DataLoader from .npz orbital data files."""
    dataset = NPZOrbitalDataset(npz_dir, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation Dataset (held-out time range)
# ─────────────────────────────────────────────────────────────────────────────
class ValidationOrbitalDataset(Dataset):
    """Held-out validation set covering a different time window."""

    def __init__(self, tle_data: TLEData, val_orbit_seconds: float = 5600):
        self.tle = tle_data

        # Validation: different time window (second orbit)
        self.times = np.linspace(val_orbit_seconds, val_orbit_seconds * 1.5, 500)

        self.orbital_elements = np.array([
            tle_data.semi_major_axis,
            tle_data.eccentricity,
            tle_data.inclination,
            tle_data.raan,
            tle_data.omega,
            tle_data.mean_anomaly,
            tle_data.mean_motion_rad_s,
        ], dtype=np.float32)

        from models.orbital_physics import OrbitalElements, propagate_j2
        kep = OrbitalElements(*self.orbital_elements)

        positions, velocities = [], []
        for t in self.times:
            pos, vel = propagate_j2(kep, t)
            positions.append(pos)
            velocities.append(vel)

        self.gt_positions = np.array(positions, dtype=np.float32)
        self.gt_velocities = np.array(velocities, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, idx: int):
        t = torch.tensor([[self.times[idx]]], dtype=torch.float32)
        orbital_elems = torch.tensor(self.orbital_elements, dtype=torch.float32).unsqueeze(0)
        position = torch.tensor(self.gt_positions[idx], dtype=torch.float32)
        velocity = torch.tensor(self.gt_velocities[idx], dtype=torch.float32)
        state = torch.cat([position, velocity])
        return t, orbital_elems, state


if __name__ == "__main__":
    from models.orbital_physics import generate_synthetic_tle

    line1, line2 = generate_synthetic_tle(400, inclination_deg=53.0)
    tle = parse_tle(line1, line2)
    print(f"Satellite: {tle.catalog_num}, a={tle.semi_major_axis:.2f} km, e={tle.eccentricity:.4f}")
    print(f"  i={math.degrees(tle.inclination):.2f}°, n={tle.mean_motion:.4f} rev/day")

    dataset = OrbitalDataset(tle, num_samples=1000)
    t, oe, state = dataset[0]
    print(f"\nSample: t={t.item():.2f}s, state shape={state.shape}")
    print(f"  Position: {state[:3].numpy()} km")
    print(f"  Velocity: {state[3:].numpy()} km/s")

    loader = create_dataloader(tle, batch_size=32)
    for batch_t, batch_oe, batch_state in loader:
        print(f"\nBatch: t={batch_t.shape}, oe={batch_oe.shape}, state={batch_state.shape}")
        break

    print("\n✓ TLE data module smoke test passed")