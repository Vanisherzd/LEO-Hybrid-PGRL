import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUT_CSV = os.path.join("data", "tdma_schedule_f5.csv")

def plot_doppler():
    if not os.path.exists(OUTPUT_CSV):
        print("Schedule not found.")
        return
        
    df = pd.read_csv(OUTPUT_CSV)
    if df.empty:
        print("Empty schedule.")
        return

    # Filter for first pass if multiple
    # extract Phase ID from Slot_ID "P1_..."
    df['Pass_ID'] = df['Slot_ID'].apply(lambda x: x.split('_')[0])
    
    unique_passes = df['Pass_ID'].unique()
    print(f"Found passes: {unique_passes}")
    
    for pid in unique_passes:
        pass_df = df[df['Pass_ID'] == pid]
        
        plt.figure(figsize=(10, 6))
        
        # Dual axis: Doppler and Elevation
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        l1 = ax1.plot(pass_df['Rel_Time_s'], pass_df['Doppler_kHz'], 'b-', label='Doppler (kHz)')
        l2 = ax2.plot(pass_df['Rel_Time_s'], pass_df['Elevation'], 'g--', label='Elevation (deg)')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Doppler Shift (kHz)', color='b')
        ax2.set_ylabel('Elevation (deg)', color='g')
        
        ax1.grid(True, linestyle=':')
        
        # Zero crossing line
        ax1.axhline(0, color='k', linewidth=0.5)
        
        plt.title(f"Doppler S-Curve Analysis ({pid}) - S-Band 2.2GHz")
        
        # Legend
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper right')
        
        save_path = os.path.join("plots", f"doppler_curve_{pid}.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()

if __name__ == "__main__":
    plot_doppler()
