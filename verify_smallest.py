import cv2

haystack = cv2.imread("candidate_0.png", cv2.IMREAD_COLOR)
template = cv2.imread("candidate_text_crop2.png", cv2.IMREAD_COLOR)

result = cv2.matchTemplate(haystack, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)
print(f"Template match in candidate_0.png: confidence={max_val:.3f}, location={max_loc}")
print(f"Expected location: (45, 10)")

# Save debug
vis = haystack.copy()
cv2.rectangle(vis, max_loc, (max_loc[0]+template.shape[1], max_loc[1]+template.shape[0]), (0,255,0), 2)
cv2.imwrite("verify_smallest_template.png", vis)
print("Saved verify_smallest_template.png")
