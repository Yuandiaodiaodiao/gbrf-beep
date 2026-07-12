from PIL import Image

img = Image.open("screenshot.png")
w, h = img.size
print(f"Full size: {w}x{h}")

# Define candidate regions to search for "回旋枪"
candidates = [
    (0.70, 0.72, 0.85, 0.82),
    (0.72, 0.75, 0.88, 0.85),
    (0.68, 0.70, 0.86, 0.80),
    (0.75, 0.68, 0.90, 0.78),
]

for i, (x1, y1, x2, y2) in enumerate(candidates):
    left = int(w * x1)
    top = int(h * y1)
    right = int(w * x2)
    bottom = int(h * y2)
    crop = img.crop((left, top, right, bottom))
    crop.save(f"candidate_{i}.png")
    print(f"Saved candidate_{i}.png: {left},{top},{right},{bottom}")
