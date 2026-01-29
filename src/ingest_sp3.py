import os
import glob
import numpy as np
from src.data.sp3_loader import SP3Loader
from src.utils import Normalizer

RAW_DIR = os.path.join("raw_data", "sp3")
OUTPUT_FILE = os.path.join("data", "precise_training_data.npz")

def ingest():
    files = glob.glob(os.path.join(RAW_DIR, "*.sp3"))
    if not files:
        print(f"No .sp3 files found in {RAW_DIR}")
        print("Please place .sp3 files in 'raw_data/sp3/'")
        return

    all_t = []
    all_states = []
    
    # Process first file for now (or concat multiple)
    # Assuming chronological order or single satellite track
    f = files[0] 
    
    loader = SP3Loader(f)
    # Auto-detect satellite? Usually SP3 has many. 
    # For Formosat-7/COSMIC-2, IDs are usually L50-L55? 
    # Let's try to detect or ask user. 
    # For PoC, let's just parse whatever is there or default to 'L50' if widely used.
    # Or just grab the first one found.
    loader.parse() 
    
    if len(loader.positions) == 0:
        print("No data parsed.")
        return
        
    eci_r, eci_v, t_sec = loader.to_eci()
    
    # Combine state
    states = np.hstack([eci_r, eci_v])
    
    # Normalize
    normalizer = Normalizer()
    t_norm = normalizer.normalize_time(t_sec)
    states_norm = normalizer.normalize_state(states)
    
    # Save
    start_epoch = loader.times[0].timestamp()
    np.savez(OUTPUT_FILE, t=t_norm, states=states_norm, t_raw=t_sec, states_raw=states, start_epoch=start_epoch)
    print(f"Saved {len(t_norm)} points to {OUTPUT_FILE} (Start Epoch: {start_epoch})")
    print(f"Velocity Check (Mean): {np.mean(np.linalg.norm(eci_v, axis=1)):.4f} km/s (Should be ~7.6)")

if __name__ == "__main__":
    ingest()
