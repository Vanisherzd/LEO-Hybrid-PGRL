import torch
import numpy as np
import pandas as pd
import datetime
import os
from src.utils import Normalizer
from src.models.neural_force import OrbitalForceNet
from src.physics.ode_solver import integrate_trajectory
from sgp4.api import jday

# --- Helpers ---
def gstime_manual(jd, fr):
    """
    Calculate GMST (actually Earth Rotation Angle) in radians.
    IAU 2000 definition of ERA is sufficient for our precision.
    """
    du = jd - 2451545.0
    tut1 = du + fr
    theta_rad = (2 * np.pi) * (0.7790572732640 + 1.00273781191135448 * tut1)
    theta_rad = theta_rad % (2 * np.pi)
    return theta_rad

# --- Configuration ---
GROUND_STATION = {
    "name": "TASA/TACC (Hsinchu)",
    "lat": 24.78,    # Degrees North
    "lon": 120.99,   # Degrees East
    "alt": 0.050,    # km (50m)
    "min_el": 10.0   # Degrees (Mask)
}

MODEL_PATH = os.path.join("weights", "f5_neural_ode_v6.pth")
DATA_PATH = os.path.join("data", "real_training_data.npz")
OUTPUT_CSV = os.path.join("data", "tdma_schedule_f5.csv")

# --- Coordinate Transforms ---
def get_ground_state_eci(t_array, epoch_timestamp):
    """
    Compute Ground Station Position and Velocity in ECI frame.
    """
    # 1. Get constant ECEF position
    lat_rad = np.radians(GROUND_STATION["lat"])
    lon_rad = np.radians(GROUND_STATION["lon"])
    alt_km = GROUND_STATION["alt"]
    
    a = 6378.137
    f = 1/298.257223563
    e2 = 2*f - f*f
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    x_obs = (N + alt_km) * np.cos(lat_rad) * np.cos(lon_rad)
    y_obs = (N + alt_km) * np.cos(lat_rad) * np.sin(lon_rad)
    z_obs = (N * (1 - e2) + alt_km) * np.sin(lat_rad)
    
    r_ecef = np.array([x_obs, y_obs, z_obs]) # Shape (3,)
    
    # 2. Rotate to ECI at each time step
    r_ground_eci = []
    v_ground_eci = []
    
    WE = 7.2921151467e-5 # Earth rotation rate (rad/s)
    
    epoch_dt = datetime.datetime.fromtimestamp(epoch_timestamp, tz=datetime.timezone.utc)
    
    for t_sec in t_array:
        curr_dt = epoch_dt + datetime.timedelta(seconds=float(t_sec))
        sec = curr_dt.second + curr_dt.microsecond / 1e6
        jd, fr = jday(curr_dt.year, curr_dt.month, curr_dt.day,
                      curr_dt.hour, curr_dt.minute, sec)
        gst = gstime_manual(jd, fr)
        
        # Rotation Matrix ECEF -> ECI (Inverse of ECI->ECEF)
        # R = [cos -sin 0; sin cos 0; 0 0 1]
        cos_g = np.cos(gst)
        sin_g = np.sin(gst)
        
        # Position Rotation
        rx = cos_g * x_obs - sin_g * y_obs
        ry = sin_g * x_obs + cos_g * y_obs
        rz = z_obs
        
        r_ground_eci.append([rx, ry, rz])
        
        # Velocity Rotation: v = omega x r
        # wxr = [-wy*rz + wz*ry, -wz*rx + wx*rz, -wx*ry + wy*rx]
        # w = [0, 0, WE]
        # wxr = [-WE*ry, WE*rx, 0]
        # NOTE: This is velocity in INERTIAL frame. 
        # Since r is changing due to rotation, dr/dt = w x r.
        
        vx = -WE * ry
        vy = WE * rx
        vz = 0.0
        
        v_ground_eci.append([vx, vy, vz])
        
    return np.array(r_ground_eci), np.array(v_ground_eci)

def calculate_doppler(r_sat, v_sat, r_ground, v_ground):
    """
    Calculate Range Rate and Doppler Shift.
    Carrier: 2.2 GHz (S-Band)
    """
    FC = 2.2e9 # Hz
    C = 299792.458 # km/s
    
    # Relative State
    r_rel = r_sat - r_ground
    v_rel = v_sat - v_ground
    
    # Range
    dist = np.linalg.norm(r_rel, axis=1)
    
    # Range Rate (dot product)
    # (r . v) / |r|
    # Broadcasting dot product: sum(a*b, axis=1)
    dot = np.sum(r_rel * v_rel, axis=1)
    range_rate = dot / dist # km/s
    
    # Doppler Shift formula: f_d = - (v_r / c) * f_c
    # approaching (negative rate) -> positive shift
    doppler_hz = -(range_rate / C) * FC
    doppler_khz = doppler_hz / 1000.0
    
    return doppler_khz

def eci2ecef_accurate(r_eci, t_array, epoch_timestamp):
    """
    Convert ECI (TEME) to ECEF using sgp4 gstime.
    """
    x_ecef, y_ecef, z_ecef = [], [], []
    
    epoch_dt = datetime.datetime.fromtimestamp(epoch_timestamp, tz=datetime.timezone.utc)
    
    for i, t_sec in enumerate(t_array):
        curr_dt = epoch_dt + datetime.timedelta(seconds=float(t_sec))
        sec = curr_dt.second + curr_dt.microsecond / 1e6
        jd, fr = jday(curr_dt.year, curr_dt.month, curr_dt.day,
                      curr_dt.hour, curr_dt.minute, sec)
        gst = gstime_manual(jd, fr)
        
        cos_g = np.cos(gst)
        sin_g = np.sin(gst)
        
        rx = r_eci[i, 0]
        ry = r_eci[i, 1]
        rz = r_eci[i, 2]
        
        tx = cos_g * rx + sin_g * ry
        ty = -sin_g * rx + cos_g * ry
        tz = rz
        
        x_ecef.append(tx)
        y_ecef.append(ty)
        z_ecef.append(tz)
        
    return np.stack([x_ecef, y_ecef, z_ecef], axis=1)

def ecef2enu(r_ecef, lat_deg, lon_deg):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    
    a = 6378.137
    f = 1/298.257223563
    e2 = 2*f - f*f
    
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    x_obs = (N + GROUND_STATION["alt"]) * np.cos(lat_rad) * np.cos(lon_rad)
    y_obs = (N + GROUND_STATION["alt"]) * np.cos(lat_rad) * np.sin(lon_rad)
    z_obs = (N * (1 - e2) + GROUND_STATION["alt"]) * np.sin(lat_rad)
    
    dx = r_ecef[:, 0] - x_obs
    dy = r_ecef[:, 1] - y_obs
    dz = r_ecef[:, 2] - z_obs
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    E = -sin_lon * dx + cos_lon * dy
    N = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    U = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return np.stack([E, N, U], axis=1)

def enu2azel(enu):
    E = enu[:, 0]
    N = enu[:, 1]
    U = enu[:, 2]
    
    horiz_dist = np.sqrt(E**2 + N**2)
    rng = np.sqrt(E**2 + N**2 + U**2)
    el = np.degrees(np.arctan2(U, horiz_dist))
    az = np.degrees(np.arctan2(E, N)) % 360.0
    
    return az, el, rng

def run_scheduler():
    print(f"--- TDMA Scheduler: {GROUND_STATION['name']} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = np.load(DATA_PATH)
    t_norm = data['t']
    states_norm = data['states']
    
    if 'start_epoch' not in data:
        print("Error: 'start_epoch' not found.")
        return
    epoch_ts = float(data['start_epoch'])
    
    r0_v0_norm = torch.tensor(states_norm[0], dtype=torch.float32).unsqueeze(0).to(device)
    model = OrbitalForceNet(input_dim=6, output_dim=3, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    normalizer = Normalizer()
    total_duration_sec = 1000 * 60 # 1000 minutes
    dt = 5.0
    steps = int(total_duration_sec / dt)
    
    print(f"Predicting orbit for {steps} steps ({total_duration_sec} s)...")
    with torch.no_grad():
        dt_norm = dt / normalizer.TU
        pred_traj_norm = integrate_trajectory(r0_v0_norm, dt_norm, model, steps=steps)
        
    pred_traj_norm = pred_traj_norm.squeeze(0).cpu().numpy()
    pred_states_km = normalizer.denormalize_state(torch.tensor(pred_traj_norm)).numpy()
    
    t_sec = np.arange(steps + 1) * dt
    r_eci = pred_states_km[:, 0:3]
    v_eci = pred_states_km[:, 3:6]
    
    print("Computing Ground Station State in ECI...")
    r_ground, v_ground = get_ground_state_eci(t_sec, epoch_ts)
    
    print("Calculate Doppler...")
    doppler_khz = calculate_doppler(r_eci, v_eci, r_ground, v_ground)
    
    print("Coordinates: ECEF -> ENU...")
    # existing coordinate logic
    r_ecef = eci2ecef_accurate(r_eci, t_sec, epoch_ts)
    enu = ecef2enu(r_ecef, GROUND_STATION["lat"], GROUND_STATION["lon"])
    az, el, rng = enu2azel(enu)
    
    print(f"Scanning for visible passes (Mask > {GROUND_STATION['min_el']} deg)...")
    schedule = []
    slot_duration = 5
    in_pass = False
    pass_id = 0
    
    for i in range(len(t_sec)):
        if el[i] >= GROUND_STATION["min_el"]:
             if not in_pass:
                in_pass = True
                pass_id += 1
                print(f"  [Pass {pass_id}] AOS at +{t_sec[i]:.0f}s")

             if t_sec[i] % slot_duration == 0:
                slot_id = f"P{pass_id}_{int(t_sec[i])}"
                schedule.append({
                    "Slot_ID": slot_id,
                    "Rel_Time_s": t_sec[i],
                    "Azimuth": round(az[i], 2),
                    "Elevation": round(el[i], 2),
                    "Range_km": round(rng[i], 2),
                    "Doppler_kHz": round(doppler_khz[i], 3)
                })
        else:
            if in_pass:
                in_pass = False
                print(f"  [Pass {pass_id}] LOS at +{t_sec[i]:.0f}s")
    
    if schedule:
        df = pd.DataFrame(schedule)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nGenerated {len(df)} slots for {pass_id} passes.")
        print(f"Sample with Doppler: \n{df.head()}")
    else:
        print(f"\nNo visible passes over Taiwan (Mask > {GROUND_STATION['min_el']} deg).")

if __name__ == "__main__":
    run_scheduler()
