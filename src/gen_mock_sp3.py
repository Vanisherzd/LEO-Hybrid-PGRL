import numpy as np
import datetime
import os

OUTPUT_FILE = "raw_data/sp3/mock_large.sp3"

def generate_mock():
    # Orbit Parameters (LEO)
    a = 7000.0 # km
    mu = 398600.4418
    n = np.sqrt(mu / a**3) # Mean motion (rad/s)
    
    t0 = datetime.datetime(2023, 10, 20, 0, 0, 0)
    dt_sec = 60 # Seconds
    points = 600 # 10 hours
    
    # Ensure dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Generating Keplerian Orbit: a={a}km, Period={2*np.pi/n:.1f}s")
    
    with open(OUTPUT_FILE, "w") as f:
        # Header (Minimal SP3)
        f.write(f"#aP2023 10 20 00 00  {points}    1\n")
        f.write("## 1\n")
        f.write("+  1    1\n")
        
        for i in range(points):
            # Time
            dt = i * dt_sec
            t = t0 + datetime.timedelta(seconds=dt)
            
            # True Anomaly (Circular)
            M = n * dt
            
            # Position (Inertial, approx)
            x = a * np.cos(M)
            y = a * np.sin(M)
            z = 0.0 # Equatorial
            
            # SP3 Body Section
            # *  2023 10 20 00 00  0.00000000
            f.write(f"*  {t.year} {t.month:>2} {t.day:>2} {t.hour:>2} {t.minute:>2}  {t.second:0<10.8f}\n")
            
            # Position Line (km)
            f.write(f"P  1   {x:14.6f}   {y:14.6f}   {z:14.6f}  99.999999\n")
            
    print(f"Generated {OUTPUT_FILE} with {points} points.")

if __name__ == "__main__":
    generate_mock()
