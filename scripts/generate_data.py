"""
Generate Synthetic Orbital Data — for training and testing LEO-PINN
=====================================================================
Uses J2-perturbed propagator (not SGP4) to produce high-fidelity ground
truth trajectories. Writes NumPy archives directly, bypassing TLE parsing.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.orbital_physics import OrbitalElements, propagate_j2, R_EARTH, MU_EARTH
import numpy as np


def generate_satellite_trajectory(
    a_km: float,
    e: float,
    i_rad: float,
    Omega_rad: float,
    omega_rad: float,
    M0_rad: float,
    n_rad_s: float,
    num_orbits: int = 3,
    samples_per_orbit: int = 200,
) -> tuple:
    """Generate trajectory using J2 propagator as ground truth.

    Returns:
        times: (N,) seconds since epoch
        positions: (N, 3) km ECI
        velocities: (N, 3) km/s ECI
        orbital_elements: (7,) [a, e, i, Omega, omega, M0, n]
    """
    kep = OrbitalElements(a_km, e, i_rad, Omega_rad, omega_rad, M0_rad, n_rad_s)

    # One orbit period
    T_orbit = 2 * np.pi / n_rad_s
    total_time = num_orbits * T_orbit

    times = np.linspace(0, total_time, num_orbits * samples_per_orbit)

    positions, velocities = [], []
    for t in times:
        pos, vel = propagate_j2(kep, t)
        positions.append(pos)
        velocities.append(vel)

    orbital_elements = np.array([a_km, e, i_rad, Omega_rad, omega_rad, M0_rad, n_rad_s], dtype=np.float32)

    return times.astype(np.float32), np.array(positions, dtype=np.float32), np.array(velocities, dtype=np.float32), orbital_elements


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Orbit parameters
    a = R_EARTH + args.altitude_km
    n = np.sqrt(MU_EARTH / a**3)  # rad/s

    # Inclinations to generate
    inclinations = [float(x) for x in args.inclinations.split(",")]

    total_sats = args.num_satellites * len(inclinations)
    print(f"Generating {total_sats} satellite trajectories")
    print(f"  altitude: {args.altitude_km} km, a={a:.3f} km, n={n:.6f} rad/s")
    print(f"  inclinations: {inclinations}")
    print(f"  orbits per sat: {args.num_orbits}")
    print(f"  output: {args.output_dir}")

    for sat_id in range(args.num_satellites):
        for inc_deg in inclinations:
            i_rad = np.radians(inc_deg)
            Omega = np.radians(np.random.uniform(0, 360)) if args.randomize_angles else 0.0
            omega = np.radians(np.random.uniform(0, 360)) if args.randomize_angles else 0.0
            M0 = np.radians(np.random.uniform(0, 360)) if args.randomize_angles else 0.0

            times, positions, velocities, orbital_elements = generate_satellite_trajectory(
                a, args.eccentricity, i_rad, Omega, omega, M0, n,
                num_orbits=args.num_orbits,
                samples_per_orbit=args.samples_per_orbit,
            )

            filename = f"sat_{sat_id:04d}_i{inc_deg:.0f}.npz"
            filepath = os.path.join(args.output_dir, filename)
            np.savez_compressed(
                filepath,
                times=times,
                positions=positions,
                velocities=velocities,
                orbital_elements=orbital_elements,
            )

            print(f"  {filepath}: a={a:.2f}km, e={args.eccentricity:.4f}, "
                  f"i={inc_deg:.1f}°, orbits={args.num_orbits}, N={len(times)}")

    print(f"\nGenerated {total_sats} files in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic orbital training data")
    parser.add_argument("--num_satellites", type=int, default=20)
    parser.add_argument("--altitude_km", type=float, default=400.0)
    parser.add_argument("--inclinations", type=str, default="53.0,72.0,98.0")
    parser.add_argument("--eccentricity", type=float, default=0.001)
    parser.add_argument("--num_orbits", type=int, default=3)
    parser.add_argument("--samples_per_orbit", type=int, default=200)
    parser.add_argument("--randomize_angles", action="store_true")
    parser.add_argument("--output_dir", type=str, default="data/tle/")
    args = parser.parse_args()
    main(args)