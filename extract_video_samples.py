#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_video_samples.py

Extract 200 positive and 200 negative training samples from a gameplay video.

Positive samples: frames where the ONNX classifier predicts "精准收招" with the
highest probability.
Negative samples: frames where the classifier predicts the background / other UI
with the lowest probability.

The script samples frames from the video, crops the same bottom-right search
region used by the live monitor, runs the existing squeezenet classifier, and
copies the top/bottom 200 scored frames into separate folders.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# Same relative region used by monitor.py / monitor_onnx.py
SEARCH_REGION = (0.7104, 0.7222, 0.7794, 0.8148)
DEFAULT_ONNX_PATH = "jzsz_classifier_squeezenet.onnx"
DEFAULT_POSITIVE_DIR = "positive_samples"
DEFAULT_NEGATIVE_DIR = "negative_samples"
DEFAULT_SAMPLE_INTERVAL = 5
DEFAULT_POSITIVE_COUNT = 200
DEFAULT_NEGATIVE_COUNT = 200

# Classifier normalization (must match training in train_classifier.py / monitor_onnx.py)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224


def get_search_bbox(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) for the search region."""
    left = int(frame_width * SEARCH_REGION[0])
    top = int(frame_height * SEARCH_REGION[1])
    right = int(frame_width * SEARCH_REGION[2])
    bottom = int(frame_height * SEARCH_REGION[3])
    return left, top, right, bottom


def load_onnx_session(onnx_path: str) -> ort.InferenceSession:
    if not Path(onnx_path).exists():
        print(f"[ERROR] ONNX model not found: {onnx_path}")
        sys.exit(1)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"Loaded ONNX model: {onnx_path}")
    return session


def classify_region(region_bgr: np.ndarray, session: ort.InferenceSession, input_name: str) -> tuple[float, int]:
    """Return positive probability and predicted class for the cropped region."""
    region_rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(region_rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    input_tensor = np.expand_dims(arr, axis=0).astype(np.float32)

    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    positive_prob = float(probs[1])
    predicted_class = int(np.argmax(logits))
    return positive_prob, predicted_class


def extract_samples(
    video_path: str,
    onnx_path: str,
    positive_dir: str,
    negative_dir: str,
    sample_interval: int,
    positive_count: int,
    negative_count: int,
):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # Prepare output folders
    for d in (positive_dir, negative_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    session = load_onnx_session(onnx_path)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    left, top, right, bottom = get_search_bbox(frame_width, frame_height)
    region_w = right - left
    region_h = bottom - top

    print(f"Video: {video_path.name}")
    print(f"  Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}, Frames: {total_frames}")
    print(f"  Search region: ({left}, {top}, {right}, {bottom}) => {region_w}x{region_h}")
    print(f"  Sampling every {sample_interval} frames, classifier input {IMG_SIZE}x{IMG_SIZE}")

    scored_regions = []  # list of (positive_prob, frame_index, predicted_class, cropped_region_rgb)
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            region_bgr = frame[top:bottom, left:right]
            positive_prob, predicted_class = classify_region(region_bgr, session, input_name)
            region_rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
            scored_regions.append((positive_prob, frame_idx, predicted_class, region_rgb))
            processed += 1

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  processed {frame_idx}/{total_frames} frames, sampled {processed}...")

    cap.release()
    print(f"Total sampled frames: {len(scored_regions)}")

    if len(scored_regions) < positive_count + negative_count:
        print(
            f"[WARNING] Not enough sampled frames ({len(scored_regions)}) for "
            f"{positive_count}+{negative_count} samples. Will save all available."
        )

    # Sort by positive probability (descending)
    scored_regions.sort(key=lambda x: x[0], reverse=True)

    positives = scored_regions[:positive_count]
    negatives = scored_regions[-negative_count:] if len(scored_regions) >= negative_count else scored_regions[::-1]

    def save_samples(samples, output_dir, prefix):
        for i, (prob, frame_idx, pred_class, region_rgb) in enumerate(samples):
            img = Image.fromarray(region_rgb)
            filename = f"{prefix}_{i+1:04d}_frame{frame_idx:05d}_prob{prob:.3f}_pred{pred_class}.jpg"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath, "JPEG", quality=92)
        print(f"Saved {len(samples)} {prefix} samples to {output_dir}/")

    save_samples(positives, positive_dir, "positive")
    save_samples(negatives, negative_dir, "negative")

    if positives and negatives:
        print(f"\nClassifier probability ranges:")
        print(f"  Positive: {positives[-1][0]:.3f} ~ {positives[0][0]:.3f}")
        print(f"  Negative: {negatives[0][0]:.3f} ~ {negatives[-1][0]:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract positive/negative training samples from a gameplay video using the ONNX classifier."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--onnx", default=DEFAULT_ONNX_PATH, help="Path to ONNX classifier model")
    parser.add_argument("--positive-dir", default=DEFAULT_POSITIVE_DIR, help="Output folder for positive samples")
    parser.add_argument("--negative-dir", default=DEFAULT_NEGATIVE_DIR, help="Output folder for negative samples")
    parser.add_argument("--interval", type=int, default=DEFAULT_SAMPLE_INTERVAL, help="Process every Nth frame")
    parser.add_argument("--positive-count", type=int, default=DEFAULT_POSITIVE_COUNT, help="Number of positive samples")
    parser.add_argument("--negative-count", type=int, default=DEFAULT_NEGATIVE_COUNT, help="Number of negative samples")
    return parser


def main():
    args = build_parser().parse_args()
    extract_samples(
        video_path=args.video,
        onnx_path=args.onnx,
        positive_dir=args.positive_dir,
        negative_dir=args.negative_dir,
        sample_interval=args.interval,
        positive_count=args.positive_count,
        negative_count=args.negative_count,
    )


if __name__ == "__main__":
    main()
