from PIL import Image

img = Image.open("interesting_frames/frame_053320_075.jpg").convert("RGB")

# Manual crop for "精准收招" text based on visual inspection
# Frame size is 265x200
x1, y1, x2, y2 = 30, 65, 235, 135
cropped = img.crop((x1, y1, x2, y2))
cropped.save("candidate_jzsz.png")
print(f"Saved candidate_jzsz.png: {cropped.size}")

vis = img.copy()
from PIL import ImageDraw
draw = ImageDraw.Draw(vis)
draw.rectangle((x1, y1, x2, y2), outline="green", width=2)
vis.save("candidate_jzsz_debug.jpg")
print(f"Saved candidate_jzsz_debug.jpg")
