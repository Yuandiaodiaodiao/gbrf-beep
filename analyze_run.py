import os
import re
import glob
from PIL import Image

def analyze_run(output_dir="missing_frames"):
    files = sorted(glob.glob(os.path.join(output_dir, "missing_*.jpg")))
    if not files:
        print("No missing frames found.")
        return

    print(f"Total missing frames saved: {len(files)}")
    
    # Get total size
    total_size = sum(os.path.getsize(f) for f in files)
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    
    # Check first and last frame dimensions
    if files:
        first = Image.open(files[0])
        print(f"Frame dimensions: {first.size}")

if __name__ == "__main__":
    analyze_run()
