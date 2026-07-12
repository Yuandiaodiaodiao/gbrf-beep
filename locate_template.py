import cv2
import numpy as np

full = cv2.imread("screenshot.png", cv2.IMREAD_COLOR)
template = cv2.imread("candidate_0.png", cv2.IMREAD_COLOR)

h, w = full.shape[:2]
th, tw = template.shape[:2]

result = cv2.matchTemplate(full, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)

x, y = max_loc
print(f"Screen size: {w}x{h}")
print(f"Template size: {tw}x{th}")
print(f"Match confidence: {max_val:.3f}")
print(f"Template top-left: ({x}, {y})")
print(f"Template bottom-right: ({x + tw}, {y + th})")

# Add a small padding around the template for the detection region
pad = 20
left = max(0, x - pad)
top = max(0, y - pad)
right = min(w, x + tw + pad)
bottom = min(h, y + th + pad)
print(f"\nTight detection region (with {pad}px padding):")
print(f"  Absolute: ({left}, {top}) - ({right}, {bottom}), size {right - left}x{bottom - top}")
print(f"  Relative: ({left/w:.4f}, {top/h:.4f}, {right/w:.4f}, {bottom/h:.4f})")

# Optional: save a visualization
vis = full.copy()
cv2.rectangle(vis, (left, top), (right, bottom), (0, 255, 0), 2)
cv2.rectangle(vis, (x, y), (x + tw, y + th), (0, 0, 255), 2)
cv2.imwrite("detect_region_debug.png", vis)
print("\nSaved visualization: detect_region_debug.png")
