import time
import cv2
import mss
import numpy as np
import sys
import argparse
import os
from PIL import Image
from datetime import datetime

# Configuration
TEMPLATE_PATH = "candidate_jzsz.png"  # template of "精准收招"
MATCH_THRESHOLD = 0.80                 # confidence threshold (0-1)
SEARCH_REGION = (0.7104, 0.7222, 0.7794, 0.8148)  # text region + 50px extra to the right
DEFAULT_INTERVAL_MS = 100                    # default screenshot interval in milliseconds

def detect_once(sct, left, top, width, height, template):
    img = sct.grab({"left": left, "top": top, "width": width, "height": height})
    haystack = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
    result = cv2.matchTemplate(haystack, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    found = max_val >= MATCH_THRESHOLD
    return found, max_val, max_loc, img

def save_frame(img, output_dir, timestamp, prefix="frame", quality=85):
    pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
    filename = f"{prefix}_{timestamp.strftime('%H%M%S_%f')[:-3]}.jpg"
    filepath = os.path.join(output_dir, filename)
    pil_img.save(filepath, "JPEG", quality=quality)
    return filepath

parser = argparse.ArgumentParser()
parser.add_argument("--once", action="store_true", help="Run one detection and exit")
parser.add_argument("--record-missing", action="store_true", help="Save every frame when text is not found")
parser.add_argument("--record-found", action="store_true", help="Save every frame when text is found")
parser.add_argument("--record-all", action="store_true", help="Save every captured frame")
parser.add_argument("--output-dir", default="frames", help="Directory to save frames")
parser.add_argument("--duration", type=int, default=0, help="Run for N seconds then stop (0 = infinite)")
parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS, help="Frame interval in milliseconds")
args = parser.parse_args()

# Load template
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
if template is None:
    raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")

template_h, template_w = template.shape[:2]
print(f"Template size: {template_w}x{template_h}, threshold: {MATCH_THRESHOLD}", flush=True)

if args.record_missing or args.record_found or args.record_all:
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Will save frames to: {args.output_dir}", flush=True)

with mss.MSS() as sct:
    monitor = sct.monitors[1]
    mw, mh = monitor["width"], monitor["height"]
    left = int(mw * SEARCH_REGION[0])
    top = int(mh * SEARCH_REGION[1])
    right = int(mw * SEARCH_REGION[2])
    bottom = int(mh * SEARCH_REGION[3])
    width = right - left
    height = bottom - top
    print(f"Monitoring region: {left},{top} {width}x{height} on monitor {mw}x{mh}", flush=True)

    if args.once:
        found, max_val, max_loc, _ = detect_once(sct, left, top, width, height, template)
        status = "FOUND" if found else "NOT FOUND"
        print(f"confidence={max_val:.3f} => {status}", flush=True)
        sys.exit(0 if found else 1)

    start_time = time.time()
    frame_count = 0
    saved_count = 0
    print(f"Running every {args.interval_ms}ms. Press Ctrl+C to stop.", flush=True)
    if args.duration > 0:
        print(f"Will stop automatically after {args.duration} seconds.", flush=True)
    print("", flush=True)

    try:
        while True:
            loop_start = time.time()
            found, max_val, max_loc, img = detect_once(sct, left, top, width, height, template)
            status = "FOUND" if found else "not found"
            frame_count += 1

            if args.record_all or (found and args.record_found) or (not found and args.record_missing):
                prefix = "found" if found else "missing" if not args.record_all else "frame"
                filepath = save_frame(img, args.output_dir, datetime.now(), prefix=prefix)
                saved_count += 1

            print(f"[{time.strftime('%H:%M:%S')}] #{frame_count} confidence={max_val:.3f} => {status}", flush=True)

            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\nDuration reached ({args.duration}s). Stopping.", flush=True)
                print(f"Total frames: {frame_count}, Frames saved: {saved_count}", flush=True)
                break

            elapsed = (time.time() - loop_start) * 1000
            sleep_ms = max(0, args.interval_ms - elapsed)
            time.sleep(sleep_ms / 1000)
    except KeyboardInterrupt:
        print(f"\nStopped by user. Frames: {frame_count}, Frames saved: {saved_count}", flush=True)
