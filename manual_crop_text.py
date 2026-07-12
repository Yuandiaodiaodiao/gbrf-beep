import cv2

img = cv2.imread("candidate_0.png", cv2.IMREAD_COLOR)

# Manual crop based on visual inspection: text "回旋枪+" is roughly in x=40..260
crops = [
    (40, 0, 280, 216, "candidate_text_crop1"),  # text + some icon edge
    (45, 10, 250, 200, "candidate_text_crop2"),  # tight around text
    (40, 0, 220, 216, "candidate_text_crop3"),  # only text
]

for x1, y1, x2, y2, name in crops:
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite(f"{name}.png", cropped)
    print(f"Saved {name}.png: {cropped.shape[1]}x{cropped.shape[0]}")
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(f"{name}_debug.png", vis)
