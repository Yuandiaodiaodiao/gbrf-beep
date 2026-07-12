import glob
import os
import cv2
import numpy as np
from PIL import Image

FRAMES_DIR = "frames"
OUTPUT_DIR = "interesting_frames"

def find_interesting_frames(frames_dir=FRAMES_DIR, output_dir=OUTPUT_DIR, top_n=20):
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if len(files) < 2:
        print("Need at least 2 frames to compare.")
        return

    os.makedirs(output_dir, exist_ok=True)
    diffs = []

    prev_gray = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    for i in range(1, len(files)):
        curr_gray = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        if prev_gray is None or curr_gray is None:
            prev_gray = curr_gray
            continue
        diff = cv2.absdiff(prev_gray, curr_gray)
        score = np.mean(diff)
        diffs.append((score, files[i]))
        prev_gray = curr_gray

    diffs.sort(reverse=True, key=lambda x: x[0])
    print(f"Total frames: {len(files)}")
    print(f"Top {top_n} most different frames:")
    for score, path in diffs[:top_n]:
        print(f"  {path}: avg diff={score:.2f}")
        # Copy to interesting_frames
        basename = os.path.basename(path)
        Image.open(path).save(os.path.join(output_dir, basename))

if __name__ == "__main__":
    find_interesting_frames()
