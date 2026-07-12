#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_positive_samples.py

Check every image in positive_samples/ against the candidate_jzsz.png template,
score the match, and copy the ones that do NOT look like "精准收招" to new_data/.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path("G:/Users/qq295/WorkBuddy/自动录屏")
TEMPLATE_PATH = ROOT / "candidate_jzsz.png"
POS_DIR = ROOT / "positive_samples"
NEG_DIR = ROOT / "new_data"
NEG_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def match_score(haystack: np.ndarray, template: np.ndarray) -> float:
    """Return best cv2.matchTemplate TM_CCOEFF_NORMED score."""
    hay_gray = cv2.cvtColor(haystack, cv2.COLOR_RGB2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    result = cv2.matchTemplate(hay_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


def main() -> None:
    template_full = load_rgb(TEMPLATE_PATH)
    tpl_h, tpl_w = template_full.shape[:2]

    samples = sorted(POS_DIR.glob("*.jpg"))
    print(f"Found {len(samples)} samples in {POS_DIR}")

    scores = []
    for path in samples:
        img = load_rgb(path)
        img_h, img_w = img.shape[:2]
        # Scale template so its width fits roughly 70 px on the text band.
        # The template is 205x70 at 4K; 1440p crops are 177x134, so scale ~0.67.
        scale = 0.667
        new_w = int(round(tpl_w * scale))
        new_h = int(round(tpl_h * scale))
        resized_tpl = cv2.resize(template_full, (new_w, new_h), interpolation=cv2.INTER_AREA)

        score = match_score(img, resized_tpl)
        scores.append((path, score))

    # Sort by score ascending: the lowest scores are the most likely false positives.
    scores.sort(key=lambda x: x[1])

    report_path = ROOT / "positive_match_scores.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"path,match_score\n")
        for path, score in scores:
            f.write(f"{path.name},{score:.4f}\n")

    print(f"Saved scores to {report_path}")
    print("\nLowest 20 scores (most likely false positives):")
    for path, score in scores[:20]:
        print(f"  {score:.4f}  {path.name}")
    print("\nHighest 5 scores:")
    for path, score in scores[-5:]:
        print(f"  {score:.4f}  {path.name}")


if __name__ == "__main__":
    main()
