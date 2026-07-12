#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_jzsz_beep.py

Windows-only, region-only screen detector for "精准收招".

Continuously captures a specified screen region at ~60 FPS and emits a system beep
when the template is matched. After a beep, detection pauses for the cooldown period.

Screenshots are NOT saved automatically; use --debug only when you need a single marked image.

Usage:
    python detect_jzsz_beep.py --region left,top,right,bottom
    python detect_jzsz_beep.py --region 2688,1555,3264,1771

Press Ctrl+C to stop.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import dxcam
import numpy as np
import winsound
from PIL import Image, ImageDraw


def beep(frequency: int = 1000, duration_ms: int = 500) -> None:
    """Play a system beep sound."""
    try:
        winsound.Beep(frequency, duration_ms)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] Failed to play beep: {exc}")


def load_image(path: Path) -> np.ndarray:
    """Load an image file as an RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def match_template(haystack_gray: np.ndarray, template_gray: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Return best-match confidence and top-left location (grayscale matching)."""
    result = cv2.matchTemplate(haystack_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc  # type: ignore[return-value]


def save_debug(
    haystack: np.ndarray,
    max_loc: tuple[int, int],
    template_shape: tuple[int, int, int],
    output_path: Path,
) -> None:
    """Save a debug image with the matched rectangle drawn."""
    img = Image.fromarray(haystack)
    draw = ImageDraw.Draw(img)
    x1, y1 = max_loc
    x2 = x1 + template_shape[1]
    y2 = y1 + template_shape[0]
    draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
    img.save(output_path)
    print(f"[DEBUG] Saved result image to: {output_path}")


def test_image(args: argparse.Namespace) -> None:
    """Run a single detection against a static image file and exit."""
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    template = load_image(args.template)
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    haystack = load_image(image_path)
    haystack_gray = cv2.cvtColor(haystack, cv2.COLOR_RGB2GRAY)
    max_val, max_loc = match_template(haystack_gray, template_gray)

    print(f"Best match: confidence={max_val:.3f} at {max_loc}")
    if max_val >= args.threshold:
        print("MATCH detected! Beep!")
        beep(args.frequency, args.duration)
    else:
        print("No match above threshold.")

    if args.debug:
        save_debug(haystack, max_loc, template.shape, Path(args.debug))


def monitor_screen(args: argparse.Namespace) -> None:
    """Continuously capture the specified screen region and beep on detection."""
    template = load_image(args.template)
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    print(f"Loaded template: {args.template} ({template.shape[1]}x{template.shape[0]})")
    print(f"Threshold={args.threshold}, interval={args.interval*1000:.2f}ms (~{1/args.interval:.0f} FPS), cooldown={args.cooldown}s")

    if not args.region:
        print("[ERROR] --region is required for screen monitoring (e.g. --region 2688,1555,3264,1771)")
        sys.exit(1)

    parts = [int(p.strip()) for p in args.region.split(",")]
    if len(parts) != 4:
        print("[ERROR] --region must be in the form left,top,right,bottom")
        sys.exit(1)
    bbox: tuple[int, int, int, int] = tuple(parts)  # type: ignore[assignment]
    print(f"Searching region: {bbox}")

    try:
        camera = dxcam.create()
    except Exception as exc:
        print(f"[ERROR] Failed to create dxcam camera: {exc}")
        sys.exit(1)

    camera.start(region=bbox, video_mode=True)

    # Warm up until the first frame is available.
    frame = None
    for _ in range(30):
        frame = camera.get_latest_frame()
        if frame is not None:
            break
        time.sleep(0.01)
    if frame is None:
        print("[ERROR] dxcam did not return a frame. Is the region on a valid monitor?")
        camera.stop()
        sys.exit(1)

    pause_until = 0.0
    last_status_print = 0.0
    print("Running... Press Ctrl+C to stop.")

    try:
        while True:
            now = time.time()

            # After a beep we completely pause detection for the cooldown period.
            if now < pause_until:
                remaining = pause_until - now
                timestamp = time.strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] cooldown: {remaining:.1f}s left  ",
                    end="\r",
                    flush=True,
                )
                time.sleep(remaining)
                continue

            frame_start = time.perf_counter()
            frame = camera.get_latest_frame()
            if frame is None:
                continue

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            max_val, max_loc = match_template(frame_gray, template_gray)
            elapsed_ms = (time.perf_counter() - frame_start) * 1000
            timestamp = time.strftime("%H:%M:%S")

            if max_val >= args.threshold:
                print(f"\n[{timestamp}] MATCH: confidence={max_val:.3f} at {max_loc} ({elapsed_ms:.1f}ms)")
                beep(args.frequency, args.duration)
                pause_until = time.time() + args.cooldown

                if args.debug:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    save_debug(frame_rgb, max_loc, template.shape, Path(args.debug))

                # Jump to the next loop iteration so cooldown is handled immediately.
                continue

            # Throttle status output so the console is not flooded at 60 FPS.
            if now - last_status_print >= 0.5:
                print(
                    f"[{timestamp}] scanning... best={max_val:.3f} ({elapsed_ms:.1f}ms)  ",
                    end="\r",
                    flush=True,
                )
                last_status_print = now

            # Maintain the target frame interval by sleeping the remainder of the frame time.
            sleep_time = args.interval - (time.perf_counter() - frame_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect '精准收招' on screen and beep when found."
    )
    parser.add_argument(
        "--template",
        default="candidate_jzsz.png",
        help="Path to the template image (default: candidate_jzsz.png)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum match confidence to trigger a beep (default: 0.8)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1 / 60,
        help="Target seconds between screen captures, e.g. 0.0167 for ~60 FPS (default: 1/60)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=3.0,
        help="After a beep, pause all detection for this many seconds (default: 3.0)",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=1000,
        help="Beep frequency in Hz (default: 1000)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=500,
        help="Beep duration in milliseconds (default: 500)",
    )
    parser.add_argument(
        "--region",
        help="Restrict search to a screen region: left,top,right,bottom",
    )
    parser.add_argument(
        "--image",
        help="Test against a single image file instead of capturing the screen",
    )
    parser.add_argument(
        "--debug",
        help="Save a debug image with the matched rectangle drawn",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"[ERROR] Template not found: {template_path.absolute()}")
        sys.exit(1)

    if args.image:
        test_image(args)
    else:
        monitor_screen(args)


if __name__ == "__main__":
    main()
