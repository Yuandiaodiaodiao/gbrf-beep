import cv2
import numpy as np

img = cv2.imread("candidate_0.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute column-wise mean brightness and find text region
# Text is bright; background is dark blue
mean_bright = np.mean(gray, axis=0)

# Use threshold: columns with mean > 120 are likely text/icon
threshold = 120
text_cols = np.where(mean_bright > threshold)[0]
print(f"Columns with mean brightness > {threshold}: {text_cols[0]} - {text_cols[-1]}")

# Find the main cluster: from first bright col, extend until big drop
def find_clusters(indices):
    if len(indices) == 0:
        return []
    clusters = []
    start = indices[0]
    prev = indices[0]
    for i in indices[1:]:
        if i > prev + 1:
            clusters.append((start, prev))
            start = i
        prev = i
    clusters.append((start, prev))
    return clusters

clusters = find_clusters(text_cols)
print(f"Bright clusters: {clusters}")

# The first big cluster is likely the text "回旋枪+"
# The second cluster is the Y icon
if len(clusters) >= 2:
    text_cluster = clusters[0]
    icon_cluster = clusters[1]
    print(f"Text cluster: {text_cluster}, Icon cluster: {icon_cluster}")
else:
    text_cluster = clusters[0]
    print(f"Only one cluster: {text_cluster}")

# Save text-only crop with padding
pad = 3
x1 = max(0, text_cluster[0] - pad)
y1 = 0
x2 = min(img.shape[1], text_cluster[1] + 1 + pad)
y2 = img.shape[0]
cropped = img[y1:y2, x1:x2]
cv2.imwrite("candidate_text_only_v2.png", cropped)
print(f"Saved candidate_text_only_v2.png: {cropped.shape[1]}x{cropped.shape[0]}")

# Save debug
vis = img.copy()
cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("candidate_text_only_v2_debug.png", vis)
print(f"Text crop offset in template: ({x1}, {y1})")
