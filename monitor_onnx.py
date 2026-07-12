import time
import mss
import numpy as np
import argparse
import os
import onnxruntime as ort
from PIL import Image
from datetime import datetime

# Configuration
ONNX_PATH = "jzsz_classifier_squeezenet.onnx"
SEARCH_REGION = (0.7104, 0.7222, 0.7794, 0.8148)  # same region as before
DEFAULT_INTERVAL_MS = 16  # ~60 FPS
POSITIVE_THRESHOLD = 0.5

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ONNX session
session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def preprocess_region(img):
    """img is mss screenshot object. Returns NCHW float32 tensor."""
    # Convert from BGRA to RGB PIL Image
    pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
    # Resize to 224x224
    pil_img = pil_img.resize((224, 224), Image.BILINEAR)
    # Convert to numpy array (H, W, C), uint8 [0, 255]
    arr = np.array(pil_img).astype(np.float32) / 255.0
    # Normalize
    arr = (arr - MEAN) / STD
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    # Add batch dimension
    return np.expand_dims(arr, axis=0).astype(np.float32)

def predict(input_tensor):
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return probs[1], int(np.argmax(logits))

def save_frame(img, output_dir, timestamp, prefix="frame", quality=85):
    pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
    filename = f"{prefix}_{timestamp.strftime('%H%M%S_%f')[:-3]}.jpg"
    filepath = os.path.join(output_dir, filename)
    pil_img.save(filepath, "JPEG", quality=quality)
    return filepath

parser = argparse.ArgumentParser()
parser.add_argument("--once", action="store_true", help="Run one detection and exit")
parser.add_argument("--record-found", action="store_true", help="Save frames when text is found")
parser.add_argument("--record-all", action="store_true", help="Save every captured frame")
parser.add_argument("--output-dir", default="cnn_frames", help="Directory to save frames")
parser.add_argument("--duration", type=int, default=0, help="Run for N seconds then stop (0 = infinite)")
parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS, help="Frame interval in milliseconds")
parser.add_argument("--threshold", type=float, default=POSITIVE_THRESHOLD, help="Positive threshold (0-1)")
args = parser.parse_args()

if args.record_found or args.record_all:
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Will save frames to: {args.output_dir}", flush=True)

print(f"ONNX model: {ONNX_PATH}")
print(f"Threshold: {args.threshold}")
print(f"Interval: {args.interval_ms}ms")

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
        img = sct.grab({"left": left, "top": top, "width": width, "height": height})
        input_tensor = preprocess_region(img)
        prob, pred = predict(input_tensor)
        found = pred == 1 and prob >= args.threshold
        print(f"prob={prob:.3f} => {'FOUND' if found else 'NOT FOUND'}", flush=True)
        exit(0 if found else 1)

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
            img = sct.grab({"left": left, "top": top, "width": width, "height": height})
            input_tensor = preprocess_region(img)
            prob, pred = predict(input_tensor)
            found = pred == 1 and prob >= args.threshold
            frame_count += 1

            if args.record_all or (found and args.record_found):
                prefix = "found" if found else "frame"
                filepath = save_frame(img, args.output_dir, datetime.now(), prefix=prefix)
                saved_count += 1

            print(f"[{time.strftime('%H:%M:%S')}] #{frame_count} prob={prob:.3f} => {'FOUND' if found else 'not found'}", flush=True)

            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\nDuration reached ({args.duration}s). Stopping.", flush=True)
                print(f"Total frames: {frame_count}, Frames saved: {saved_count}", flush=True)
                break

            elapsed = (time.time() - loop_start) * 1000
            sleep_ms = max(0, args.interval_ms - elapsed)
            time.sleep(sleep_ms / 1000)
    except KeyboardInterrupt:
        print(f"\nStopped by user. Frames: {frame_count}, Frames saved: {saved_count}", flush=True)
