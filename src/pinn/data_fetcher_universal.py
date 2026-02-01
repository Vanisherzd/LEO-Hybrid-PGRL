import requests
import numpy as np
import sys
from sgp4.api import Satrec, WGS72
from sgp4.conveniences import sat_epoch_datetime
import datetime
from src.pinn.utils import Normalizer

def fetch_tle(norad_id):
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    print(f"Fetching TLE for {norad_id} from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        tle_text = response.text.strip().splitlines()
        if len(tle_text) >= 2:
            line1, line2 = None, None
            for line in tle_text:
                if line.startswith('1 '):
                    line1 = line
                elif line.startswith('2 '):
                    line2 = line
            
            if line1 and line2:
                return line1, line2
            if len(tle_text) == 2:
                return tle_text[0], tle_text[1]
            if len(tle_text) == 3:
                return tle_text[1], tle_text[2]
        
        print(f"Error: Invalid TLE format for {norad_id}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Error fetching TLE: {e}")
        sys.exit(1)

def fetch_tle_and_propagate(norad_id, duration_mins=300):
    l1, l2 = fetch_tle(norad_id)
    print(f"Line 1: {l1}")
    print(f"Line 2: {l2}")
    
    satellite = Satrec.twoline2rv(l1, l2, WGS72)
    epoch_dt = sat_epoch_datetime(satellite)
    print(f"TLE Epoch Date: {epoch_dt} (UTC)")
    
    start_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    steps = duration_mins * 60
    
    print(f"Propagating for {duration_mins} minutes ({steps} steps)...")
    
    times = []
    states = []
    
    from sgp4.api import jday
    for i in range(steps):
        current_time = start_time + datetime.timedelta(seconds=i)
        jd, fr = jday(current_time.year, current_time.month, current_time.day,
                      current_time.hour, current_time.minute, current_time.second)
        
        e, r, v = satellite.sgp4(jd, fr)
        if e != 0: continue
        
        times.append(i) # t in seconds relative to start
        states.append(list(r) + list(v))
        
    return np.array(times), np.array(states)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("norad_id", type=int, help="NORAD ID of satellite")
    parser.add_argument("--output", type=str, default=os.path.join("data", "real_data.npz"), help="Output filename")
    args = parser.parse_args()
    
    t_raw, states_raw = fetch_tle_and_propagate(args.norad_id)
    
    normalizer = Normalizer()
    t_norm = normalizer.normalize_time(t_raw)
    states_norm = normalizer.normalize_state(states_raw)
    
    np.savez(args.output, t=t_norm, states=states_norm, t_raw=t_raw, states_raw=states_raw)
    print(f"Saved {len(t_norm)} points to {args.output}")
