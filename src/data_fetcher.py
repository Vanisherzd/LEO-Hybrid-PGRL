import requests
import numpy as np
from sgp4.api import Satrec, WGS72
from sgp4.conveniences import sat_epoch_datetime
import datetime
import sys
import os

def fetch_tle(norad_id):
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    print(f"Fetching TLE from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        tle_text = response.text.strip().splitlines()
        if len(tle_text) >= 2:
            # SGP4 expects line 1 and line 2. 
            # If CelesTrak returns 3 lines (header + 2 lines), ensure we get the TLE lines.
            # Usually GP returns 0-line header, 1-line, or 3-line format.
            # Standard TLE format usually has line 1 starting with '1 ' and line 2 starting with '2 '.
            line1 = None
            line2 = None
            for line in tle_text:
                if line.startswith('1 '):
                    line1 = line
                elif line.startswith('2 '):
                    line2 = line
            
            if line1 and line2:
                return line1, line2
            else:
                # If simple 2 line response
                if len(tle_text) == 2:
                    return tle_text[0], tle_text[1]
                elif len(tle_text) == 3:
                     return tle_text[1], tle_text[2]

        print("Error: Invalid TLE format received.")
        print(response.text)
        sys.exit(1)
        
    except requests.RequestException as e:
        print(f"Error fetching TLE: {e}")
        sys.exit(1)

def propagate_orbit(line1, line2, duration_minutes=300, dt=1.0):
    satellite = Satrec.twoline2rv(line1, line2, WGS72)
    
    # Print Epoch
    epoch_dt = sat_epoch_datetime(satellite)
    print(f"TLE Epoch Date: {epoch_dt} (UTC)")
    
    # Start propagation from "Now"
    start_time = datetime.datetime.now(datetime.timezone.utc)
    # Align to full seconds for cleaner steps
    start_time = start_time.replace(microsecond=0)
    
    times = []
    states = []
    
    steps = int(duration_minutes * 60 / dt)
    print(f"Propagating for {duration_minutes} minutes ({steps} steps) starting {start_time}...")
    
    # Generate JD and fr arrays for efficiency is possible, but loop is fine for 18000 steps
    # Using sgp4.api.jday
    from sgp4.api import jday
    
    for i in range(steps):
        current_time = start_time + datetime.timedelta(seconds=i*dt)
        jd, fr = jday(current_time.year, current_time.month, current_time.day,
                      current_time.hour, current_time.minute, current_time.second)
        
        e, r, v = satellite.sgp4(jd, fr)
        
        if e != 0:
            print(f"Error generating state at step {i}: error code {e}")
            continue
            
        # r is x,y,z in km
        # v is vx,vy,vz in km/s
        # TEME frame
        
        times.append(i * dt) # t relative to start in seconds
        states.append(list(r) + list(v))

    return np.array(times), np.array(states), start_time

if __name__ == "__main__":
    from src.utils import Normalizer
    
    NORAD_ID = 42920 # Formosat-5
    
    l1, l2 = fetch_tle(NORAD_ID)
    print(f"Line 1: {l1}")
    print(f"Line 2: {l2}")
    
    t_array, states_array, start_time = propagate_orbit(l1, l2)
    
    # Normalize Data
    normalizer = Normalizer()
    print("Normalizing data...")
    t_norm = normalizer.normalize_time(t_array)
    states_norm = normalizer.normalize_state(states_array)
    
    output_file = os.path.join("data", "real_training_data.npz")
    np.savez(output_file, t=t_norm, states=states_norm, 
             t_raw=t_array, states_raw=states_array,
             start_epoch=start_time.timestamp()) # Save epoch for GMST
    
    print(f"Saved {len(t_norm)} points to {output_file} (Epoch: {start_time})")
    print("DONE_DATA_FETCH")
