import numpy as np
import datetime
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy.time import Time
from astropy import units as u
from scipy.interpolate import CubicSpline

class SP3Loader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.times = []
        self.positions = [] # km (ECEF/ITRF)
        self.velocities = [] # km/s (Derived or Read)
        self.satellite_id = None
        
    def parse(self, target_sat_id=None):
        """
        Parse SP3-c/d format.
        target_sat_id: e.g. 'L50' or 'G01'. If None, reads first sat.
        """
        print(f"Parsing SP3: {self.filepath}")
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
            
        # Header info could be parsed here if needed
        
        current_epoch = None
        
        data_pos = []
        data_times = []
        
        for line in lines:
            if line.startswith('*'):
                # Epoch line: *  2019 12 31 23 45  0.00000000
                parts = line.split()
                year = int(parts[1])
                month = int(parts[2])
                day = int(parts[3])
                hour = int(parts[4])
                minute = int(parts[5])
                sec = float(parts[6])
                current_epoch = datetime.datetime(year, month, day, hour, minute, int(sec), tzinfo=datetime.timezone.utc)
                
            elif line.startswith('P'):
                # Position line: P01 -1234.567  9876.543  1111.222 ...
                # Sat ID is char 1-4 (e.g. 'PG01', 'PL50' or just 'P  1')
                # SP3 format varies slightly. Standard is 'P<SatID>'.
                sat_id_field = line[1:4].strip()
                
                # Simple filter
                if target_sat_id and sat_id_field != target_sat_id:
                    continue
                    
                # Store data
                # SP3 positions are usually in km
                try:
                    # columns: X Y Z Clock
                    # Fixed width is safer but split works for standard SP3
                    parts = line.split()
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    
                    data_pos.append([x, y, z])
                    data_times.append(current_epoch)
                except ValueError:
                    continue
                    
        self.positions = np.array(data_pos)
        self.times = data_times
        print(f"Loaded {len(self.positions)} points.")
        
    def derive_velocity(self):
        """
        Compute velocity from position using Cubic Spline derivative.
        """
        if len(self.positions) < 4:
            print("Not enough points to derive velocity.")
            return

        # Convert datetimes to seconds from start
        t0 = self.times[0]
        t_sec = np.array([(t - t0).total_seconds() for t in self.times])
        
        # Spline for each component
        vx = CubicSpline(t_sec, self.positions[:, 0]).derivative()(t_sec)
        vy = CubicSpline(t_sec, self.positions[:, 1]).derivative()(t_sec)
        vz = CubicSpline(t_sec, self.positions[:, 2]).derivative()(t_sec)
        
        self.velocities = np.stack([vx, vy, vz], axis=1)
        print(" Velocities computed via Cubic Spline.")

    def to_eci(self):
        """
        Convert ITRS (Earth Fixed) to GCRS (Inertial J2000).
        Using high-precision Astropy transformation (downloading IERS if needed).
        """
        print("Converting ITRF -> GCRS (ECI)...")
        if len(self.positions) == 0:
            return None, None
            
        bs = 1000 # Batch size to avoid memory issues
        eci_pos = []
        eci_vel = []
        
        # We need to transform Pos and Vel together correctly?
        # Astropy transforms positions well. Velocities require representing as differentials.
        # Alternatively: Transform r_ecef -> r_eci, then re-derive velocity in ECI.
        # This avoids complex differential transformation issues in Astropy ensuring consistency.
        
        # 1. Transform Position
        t_objs = Time(self.times)
        
        # Batch processing
        for i in range(0, len(self.positions), bs):
            end = min(i+bs, len(self.positions))
            batch_pos = self.positions[i:end]
            batch_time = t_objs[i:end]
            
            itrs = ITRS(x=batch_pos[:,0]*u.km, y=batch_pos[:,1]*u.km, z=batch_pos[:,2]*u.km, obstime=batch_time)
            gcrs = itrs.transform_to(GCRS(obstime=batch_time))
            
            # Extract cartesian
            xyz = gcrs.cartesian.xyz.to(u.km).value.T
            eci_pos.append(xyz)
            
        self.eci_positions = np.vstack(eci_pos)
        
        # 2. Derive Velocity in ECI frame (Numerical diff of ECI positions)
        # This automatically handles the Coriolis/Rotation terms implicitly
        t0 = self.times[0]
        t_sec = np.array([(t - t0).total_seconds() for t in self.times])
        
        vx = CubicSpline(t_sec, self.eci_positions[:, 0]).derivative()(t_sec)
        vy = CubicSpline(t_sec, self.eci_positions[:, 1]).derivative()(t_sec)
        vz = CubicSpline(t_sec, self.eci_positions[:, 2]).derivative()(t_sec)
        
        self.eci_velocities = np.stack([vx, vy, vz], axis=1)
        
        return self.eci_positions, self.eci_velocities, t_sec

if __name__ == "__main__":
    # Test stub
    pass
