import numpy as np
import torch

class LEOCommModel:
    """
    Simulates the PHY/MAC characteristics of a LEO-to-IoT link.
    """
    def __init__(self, f_center=2.2e9, slot_duration=0.010, device='cpu'):
        self.f_center = f_center          # 2.2 GHz (S-Band)
        self.slot_duration = slot_duration # 10ms TDMA slot
        self.c = 299792.458               # Speed of light km/s
        self.device = device
        
        # Link Budget Constants
        self.k_boltzmann = 1.38e-23
        self.t_noise = 290.0              # Kelvin
        self.bw = 1e6                     # 1 MHz bandwidth
        self.gs_gain = 30.0               # dBi (Ground Station)
        self.iot_gain = 0.0               # dBi (IoT terminal)
        
    def calculate_link_metrics(self, r_sat, v_sat, r_gs, v_gs, tx_power_dbm, guard_band_ratio):
        """
        Calculates SNR, BER, and Throughput for a given state and action.
        """
        # 1. Geometry
        r_rel = r_sat - r_gs
        v_rel = v_sat - v_gs
        dist_km = np.linalg.norm(r_rel)
        
        # 2. Free Space Path Loss (FSPL)
        # FSPL(dB) = 20log10(d) + 20log10(f) + 20log10(4pi/c)
        fspl_db = 20 * np.log10(dist_km) + 20 * np.log10(self.f_center/1e3) + 20 * np.log10(4 * np.pi / self.c)
        
        # 3. Doppler Shift
        range_rate = np.dot(r_rel, v_rel) / dist_km
        f_doppler = - (range_rate / self.c) * self.f_center
        
        # 4. Received Power (dBm)
        rx_power_dbm = tx_power_dbm + self.iot_gain + self.gs_gain - fspl_db
        
        # 5. Noise Power (dBm)
        # N = k * T * B
        noise_p_w = self.k_boltzmann * self.t_noise * self.bw
        noise_p_dbm = 10 * np.log10(noise_p_w) + 30
        
        # 6. SNR (dB)
        snr_db = rx_power_dbm - noise_p_dbm
        
        # Add Doppler Sync Penalty
        # High Doppler rate increases synchronization error and reduces SNR
        sync_penalty = 10 * np.log10(1 + (np.abs(f_doppler) / 1e5)) # Rough model
        snr_db -= sync_penalty
        
        # 7. Physical Efficiency & Throughput
        # Shannon Limit-ish
        spectral_efficiency = np.maximum(0, np.log2(1 + 10**(snr_db/10)))
        
        # Guard Band Loss
        # Effective data duration = Slot * (1 - GuardBand)
        effective_duration = self.slot_duration * (1 - guard_band_ratio)
        data_rate_bps = self.bw * spectral_efficiency * (1 - guard_band_ratio)
        
        # Packet Loss Model (Probabilistic)
        # Higher Guard Band = Lower Collision Probability
        # Collision P = exp(-guard_band_ratio * 20)
        p_collision = np.exp(-guard_band_ratio * 15.0)
        
        # SNR based loss (BER approximation for BPSK-ish)
        # BER = 0.5 * erfc(sqrt(10^(snr/10)))
        eb_n0_lin = 10**(snr_db/10)
        ber = 0.5 * np.exp(-eb_n0_lin) # Very rough approximation
        p_bit_loss = 1 - (1 - ber)**1000 # 1000 bits packet
        
        success_prob = (1 - p_collision) * (1 - p_bit_loss)
        
        return {
            "dist_km": dist_km,
            "snr_db": snr_db,
            "data_rate_bps": data_rate_bps,
            "success_prob": success_prob,
            "power_consumed_j": 10**(tx_power_dbm/10) * 1e-3 * effective_duration
        }

if __name__ == "__main__":
    # Test
    comm = LEOCommModel()
    res = comm.calculate_link_metrics(
        np.array([0, 0, 7000]), np.array([7.5, 0, 0]),
        np.array([0, 0, 6378]), np.array([0, 0, 0]),
        tx_power_dbm=20, guard_band_ratio=0.1
    )
    print(res)
