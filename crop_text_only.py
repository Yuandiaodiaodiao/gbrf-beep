import cv2
import numpy as np

img = cv2.imread("candidate_0.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to get bright pixels
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

# Sum per column and row
col_sum = np.sum(binary, axis=0)
row_sum = np.sum(binary, axis=1)

# Find first and last bright column/row with some tolerance
threshold = 255  # at least one bright pixel
bright_cols = np.where(col_sum >= threshold)[0]
bright_rows = np.where(row_sum >= threshold)[0]

x1 = bright_cols[0]
x2 = bright_cols[-1]
y1 = bright_rows[0]
y2 = bright_rows[-1]
print(f"Bright pixel bounds: x={x1}-{x2}, y={y1}-{y2}, size={x2-x1+1}x{y2-y1+1}")

# Now crop only the text part "回旋枪+" (left side), excluding the Y icon
# We can find a vertical gap between text and icon by looking at column sums
# Look for a large gap after the main text cluster
max_gap = 0
gap_x = x2
for x in range(x1 + 50, x2 - 10):
    if col_sum[x] == 0 and all(col_sum[x:x+10] <= 255):
        gap_x = x
        break

print(f"Gap after text at x≈{gap_x}")

# Try different crop widths and save for inspection
crops = [
    (x1, y1, x2 + 1, y2 + 1, "candidate_all_text"),  # all bright content
    (x1, y1, min(x2 + 1, gap_x), y2 + 1, "candidate_text_only"),  # text only, no Y icon
]
for x1c, y1c, x2c, y2c, name in crops:
    pad = 3
    cx1 = max(0, x1c - pad)
    cy1 = max(0, y1c - pad)
    cx2 = min(img.shape[1], x2c + pad)
    cy2 = min(img.shape[0], y2c + pad)
    cropped = img[cy1:cy2, cx1:cx2]
    cv2.imwrite(f"{name}.png", cropped)
    print(f"Saved {name}.png: {cropped.shape[1]}x{cropped.shape[0]}, offset in template=({cx1},{cy1})")
    
    # Save debug with bounding box on original
    vis = img.copy()
    cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
    cv2.imwrite(f"{name}_debug.png", vis)
