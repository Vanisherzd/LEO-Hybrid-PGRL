import requests
import os
import shutil
import gzip
from datetime import datetime
from dotenv import load_dotenv

# Load credentials
load_dotenv()

NASA_USER = os.getenv("NASA_USER")
NASA_PASS = os.getenv("NASA_PASS")
OUTPUT_DIR = os.path.join("raw_data", "sp3")

class EarthdataDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.auth = (NASA_USER, NASA_PASS)
        # Cookie Jar is handled automatically by Session
        
    def download(self, url, filename):
        if not NASA_USER or not NASA_PASS:
            print("Error: NASA_USER/NASA_PASS not set in .env")
            return False
            
        print(f"Downloading {url}...")
        try:
            # First request might redirect to URS
            response = self.session.get(url, stream=True, allow_redirects=True)
            
            # Check if login failed (response URL might still be urs.earthdata.nasa.gov if failed)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                print(f"Saved {filename}")
                return True
            elif response.status_code == 401:
                print("Authentication failed. Check credentials.")
                return False
            else:
                print(f"Failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Download Error: {e}")
            return False

def decompress(filename):
    """Decompress .gz or .Z files"""
    if filename.endswith('.gz'):
        clean_name = filename[:-3]
        print(f"Decompressing {filename} -> {clean_name}")
        with gzip.open(filename, 'rb') as f_in:
            with open(clean_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(filename)
        return clean_name
    return filename

def fetch_cosmic2_pod(date_obj):
    """
    Example logic to fetch COSMIC-2 POD.
    URL structure varies by provider (CDDIS vs UCAR).
    Using CDDIS logic for GNSS products as placeholder.
    Real COSMIC-2 paths: https://cddis.nasa.gov/archive/gnss/products/ionex/... 
    Actually, COSMIC-2 is often CDAAC.
    Let's assume a generic CDDIS GNSS SP3 URL for demonstration or User supplied URL.
    
    For PROOF OF CONCEPT, we fetch an IGS GPS SP3 which is public (with login) and standard.
    """
    # IGS Product URL Pattern (Daily)
    # https://cddis.nasa.gov/archive/gnss/products/[wwww]/igs[wwww][d].sp3.Z
    
    # Simple algorithm to calculate GPS Week/Day
    # Using astropy or simple math
    # Placeholder: User should provide exact URL in real usage
    # or we implement full GPS time conversion.
    
    print("Logic for automatic URL construction pending specific data source selection.")
    print("Using a test URL or manual input is recommended for now.")
    pass

if __name__ == "__main__":
    # Ensure dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("--- SP3 Data Downloader ---")
    print("1. Ensure .env has credentials.")
    print("2. Modify script to target specific URLs.")
    
    dl = EarthdataDownloader()
    
    # Example URL (IGS GPS for test)
    # url = "https://cddis.nasa.gov/archive/gnss/products/2285/igs22850.sp3.Z"
    # target = os.path.join(OUTPUT_DIR, "igs22850.sp3.Z")
    # if dl.download(url, target):
    #     decompress(target)
    
    print("Downloader module ready. Import and use 'download(url, path)'.")
