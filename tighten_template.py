import cv2
import numpy as np

img = cv2.imread("candidate_0.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold: bright text/icons against dark blue background
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

# Find bounding box of bright pixels
coords = cv2.findNonZero(binary)
x, y, w, h = cv2.boundingRect(coords)
print(f"Original template size: {img.shape[1]}x{img.shape[0]}")
print(f"Bright region bounding box: x={x}, y={y}, w={w}, h={h}")

# Add a small padding
pad = 3
crop_x = max(0, x - pad)
crop_y = max(0, y - pad)
crop_w = min(img.shape[1] - crop_x, w + pad * 2)
crop_h = min(img.shape[0] - crop_y, h + pad * 2)
print(f"Crop with {pad}px padding: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

cropped = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
cv2.imwrite("candidate_tight.png", cropped)
print("Saved candidate_tight.png")

# Save debug image showing bounding box
vis = img.copy()
cv2.rectangle(vis, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (0, 255, 0), 2)
cv2.imwrite("candidate_tight_debug.png", vis)
print("Saved candidate_tight_debug.png")
