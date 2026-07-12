import cv2
import numpy as np
from PIL import Image

haystack = Image.open("interesting_frames/frame_053320_075.jpg").convert("RGB")
template = Image.open("candidate_jzsz.png").convert("RGB")

h = cv2.cvtColor(np.array(haystack), cv2.COLOR_RGB2BGR)
t = cv2.cvtColor(np.array(template), cv2.COLOR_RGB2BGR)

result = cv2.matchTemplate(h, t, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)
print(f"Max confidence: {max_val:.3f} at {max_loc}")

vis = h.copy()
cv2.rectangle(vis, max_loc, (max_loc[0]+t.shape[1], max_loc[1]+t.shape[0]), (0,255,0), 2)
cv2.imwrite("debug_jzsz_match.jpg", vis)
