import cv2
import numpy as np

img = cv2.imread("candidate_0.png", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Value channel: text is bright, background dark
v = hsv[:, :, 2]

# Threshold on V to find bright text
_, binary = cv2.threshold(v, 160, 255, cv2.THRESH_BINARY)

# Morphological closing to connect broken text strokes
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

coords = cv2.findNonZero(binary)
x, y, w, h = cv2.boundingRect(coords)
print(f"Text region after closing: x={x}, y={y}, w={w}, h={h}")

# Try to split text and icon by looking for a large horizontal gap
# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
print(f"Connected components (excluding background): {num_labels - 1}")
for i in range(1, num_labels):
    x_i, y_i, w_i, h_i, area_i = stats[i]
    print(f"  Component {i}: x={x_i}, y={y_i}, w={w_i}, h={h_i}, area={area_i}")

# Save binary debug
cv2.imwrite("candidate_binary_debug.png", binary)

# Crop a few candidates for inspection
crops = [
    (x, y, x + w, y + h, "candidate_text_hsv"),
]

# Also try cropping just the left half where text is
left_half = img[:, :img.shape[1] // 2]
cv2.imwrite("candidate_left_half.png", left_half)
print(f"Saved candidate_left_half.png: {left_half.shape[1]}x{left_half.shape[0]}")

for x1, y1, x2, y2, name in crops:
    pad = 3
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img.shape[1], x2 + pad)
    cy2 = min(img.shape[0], y2 + pad)
    cropped = img[cy1:cy2, cx1:cx2]
    cv2.imwrite(f"{name}.png", cropped)
    print(f"Saved {name}.png: {cropped.shape[1]}x{cropped.shape[0]}")
    vis = img.copy()
    cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
    cv2.imwrite(f"{name}_debug.png", vis)
