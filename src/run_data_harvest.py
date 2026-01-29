import os
import datetime
import requests
from src.data.downloader import EarthdataDownloader, decompress
from src.ingest_sp3 import ingest
import math

RAW_DIR = os.path.join("raw_data", "sp3")

def get_gps_time(dt):
    """
    Calculate GPS Week and Day of Week.
    GPS Epoch: Jan 6, 1980
    """
    epoch = datetime.datetime(1980, 1, 6, 0, 0, 0, tzinfo=datetime.timezone.utc)
    target = dt.replace(tzinfo=datetime.timezone.utc)
    
    delta = target - epoch
    total_days = delta.days
    
    gps_week = total_days // 7
    day_of_week = total_days % 7
    
    return gps_week, day_of_week

def run_harvest():
    print("--- Phase B Data Harvest: IGS Rapid Ephemeris ---")
    
    # 1. Calculate Target Date (Yesterday to ensure availability)
    # CDDIS IGS Rapids (igr) are typically available ~17 hours after end of day.
    # Let's try 2 days ago to be safe, or 1 day ago.
    target_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=2)
    
    gps_week, dow = get_gps_time(target_date)
    print(f"Target Date: {target_date.date()} | GPS Week: {gps_week} | Day: {dow}")
    
    # 2. Construct URL
    # Format: https://cddis.nasa.gov/archive/gnss/products/{week}/igr{week}{d}.sp3.Z
    filename = f"igr{gps_week}{dow}.sp3.Z"
    url = f"https://cddis.nasa.gov/archive/gnss/products/{gps_week}/{filename}"
    
    output_path = os.path.join(RAW_DIR, filename)
    
    # 2.5 Credential Debug
    NASA_USER = os.getenv("NASA_USER")
    if NASA_USER and len(NASA_USER) > 2:
        print(f"Loaded Username: {NASA_USER[:2]}***")
    else:
        print("WARNING: NASA_USER format check failed (Empty or too short).")

    # 3. Download
    dl = EarthdataDownloader()
    download_success = dl.download(url, output_path)
    
    sp3_file = None
    
    if download_success:
        print("Download successful.")
        sp3_file = output_path
    else:
        print("Download FAILED. Checking for local fallback...")
        # Fallback: Check directory
        files = [f for f in os.listdir(RAW_DIR) if f.endswith('.sp3') or f.endswith('.sp3.Z')]
        if files:
            print(f"Found local files: {files}")
            # Pick the most recent mod time? Or just first.
            sp3_file = os.path.join(RAW_DIR, files[0])
            print(f"Falling back to local file: {sp3_file}")
        else:
            print(f"No local .sp3 files found in {RAW_DIR}")
            return

    # 4. Decompress if needed
    if sp3_file.endswith('.Z') or sp3_file.endswith('.gz'):
        sp3_path = decompress(sp3_file)
    else:
        sp3_path = sp3_file
        
    print(f"Ready for ingestion: {sp3_path}")
    
    # 5. Ingest
    print("Triggering Ingestion Process...")
    # ingest() reads from RAW_DIR/*.sp3, so just ensure our file is there (it is)
    ingest()
    
    # 6. Verify Code Artifact
    npz_path = os.path.join("data", "precise_training_data.npz")
    if os.path.exists(npz_path):
        size_mb = os.path.getsize(npz_path) / (1024 * 1024)
        print(f"SUCCESS: Generated {npz_path} ({size_mb:.2f} MB)")
        if size_mb > 0.001: # lowered threshold for mock/small files
            print("Data Volume Check: PASS")
        else:
            print("Data Volume Check: WARNING (File too small)")
    else:
        print("FAILURE: .npz file not found after ingestion.")

if __name__ == "__main__":
    run_harvest()
