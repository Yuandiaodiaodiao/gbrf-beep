#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_jzsz_beep_onnx.py

基于 ONNX CNN 分类器的 "精准收招" 屏幕检测提示音脚本。

- 使用训练好的 CNN 分类器（SqueezeNet/MobileNet ONNX）比模板匹配更鲁棒。
- 自动根据屏幕分辨率计算检测区域。
- 命中后播放提示音并进入冷却，防止连续触发。
- 支持从任意工作目录启动，模型路径以脚本所在目录为基准。

Usage:
    python detect_jzsz_beep_onnx.py
    python detect_jzsz_beep_onnx.py --region 2719,1559,2993,1764
    python detect_jzsz_beep_onnx.py --threshold 0.6 --cooldown 2.0
    python detect_jzsz_beep_onnx.py --run-duration 60 --no-beep

Press Ctrl+C to stop.
"""

import argparse
import sys
import time
import winsound
from pathlib import Path
from datetime import datetime

import mss
import numpy as np
import onnxruntime as ort
from PIL import Image


# Script directory is used as the base for model/template paths so the script
# can be launched from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "jzsz_classifier_squeezenet.onnx"

# Normalized search region for the "精准收招" text in the bottom-right corner.
# This is the same region used by monitor_onnx.py.
SEARCH_REGION = (0.7104, 0.7222, 0.7794, 0.8148)

DEFAULT_INTERVAL_MS = 16
DEFAULT_THRESHOLD = 0.5
DEFAULT_COOLDOWN = 3.0
DEFAULT_FREQUENCY = 1000
DEFAULT_DURATION = 500

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def beep(frequency: int, duration_ms: int) -> None:
    """Play a system beep sound."""
    try:
        winsound.Beep(frequency, duration_ms)
    except Exception as exc:  # pragma: no cover
        print(f"\n[ERROR] Failed to play beep: {exc}")


def preprocess_region(img) -> np.ndarray:
    """Convert an mss screenshot to a normalized NCHW float32 tensor."""
    pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
    pil_img = pil_img.resize((224, 224), Image.BILINEAR)
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0).astype(np.float32)


def predict(session, input_name: str, input_tensor: np.ndarray) -> float:
    """Return the probability of the positive class (index 1)."""
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return float(probs[1])


def parse_region(arg: str) -> tuple[int, int, int, int]:
    """Parse a 'left,top,right,bottom' string into a tuple of ints."""
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 4:
        raise ValueError("region must be in the form left,top,right,bottom")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect '精准收招' using an ONNX CNN classifier and beep when found."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the ONNX model (default: jzsz_classifier_squeezenet.onnx)",
    )
    parser.add_argument(
        "--region",
        help="Override the search region: left,top,right,bottom",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Probability threshold to trigger a beep (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help=f"Target interval between frames in ms (default: {DEFAULT_INTERVAL_MS})",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN,
        help=f"Seconds to pause after a beep (default: {DEFAULT_COOLDOWN})",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=DEFAULT_FREQUENCY,
        help=f"Beep frequency in Hz (default: {DEFAULT_FREQUENCY})",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Beep duration in milliseconds (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--run-duration",
        type=int,
        default=0,
        help="Run for N seconds then stop (0 = infinite, default: 0)",
    )
    parser.add_argument(
        "--no-beep",
        action="store_true",
        help="Print detections without actually playing the beep sound",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Interval: {args.interval_ms}ms")
    print(f"Cooldown: {args.cooldown}s")
    print(f"Beep: {args.frequency}Hz / {args.duration}ms")
    if args.run_duration > 0:
        print(f"Run duration: {args.run_duration}s")
    print("")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        mw, mh = monitor["width"], monitor["height"]

        if args.region:
            left, top, right, bottom = parse_region(args.region)
        else:
            left = int(mw * SEARCH_REGION[0])
            top = int(mh * SEARCH_REGION[1])
            right = int(mw * SEARCH_REGION[2])
            bottom = int(mh * SEARCH_REGION[3])

        width = right - left
        height = bottom - top
        print(f"Monitor: {mw}x{mh}, Region: {left},{top},{right},{bottom} ({width}x{height})")
        print("Running... Press Ctrl+C to stop.\n")

        start_time = time.time()
        pause_until = 0.0
        frame_count = 0
        last_status_print = 0.0

        try:
            while True:
                now = time.time()

                # Stop after the requested run duration.
                if args.run_duration > 0 and (now - start_time) >= args.run_duration:
                    print(f"\nRun duration reached ({args.run_duration}s). Stopping.")
                    print(f"Total frames: {frame_count}")
                    break

                # After a beep we pause all detection for the cooldown period.
                if now < pause_until:
                    remaining = pause_until - now
                    if now - last_status_print >= 0.5:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] cooldown: {remaining:.1f}s left   ",
                            end="\r",
                            flush=True,
                        )
                        last_status_print = now
                    time.sleep(remaining)
                    continue

                loop_start = time.perf_counter()
                img = sct.grab({
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                })
                input_tensor = preprocess_region(img)
                prob = predict(session, input_name, input_tensor)
                found = prob >= args.threshold
                frame_count += 1

                if found:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(
                        f"\n[{timestamp}] #{frame_count} FOUND prob={prob:.3f} => BEEP",
                        flush=True,
                    )
                    if not args.no_beep:
                        beep(args.frequency, args.duration)
                    pause_until = time.time() + args.cooldown
                    last_status_print = 0.0
                else:
                    if now - last_status_print >= 0.5:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] #{frame_count} scanning... prob={prob:.3f}   ",
                            end="\r",
                            flush=True,
                        )
                        last_status_print = now

                elapsed = (time.perf_counter() - loop_start) * 1000
                sleep_ms = max(0, args.interval_ms - elapsed)
                time.sleep(sleep_ms / 1000)
        except KeyboardInterrupt:
            print("\nStopped by user.")


if __name__ == "__main__":
    main()
