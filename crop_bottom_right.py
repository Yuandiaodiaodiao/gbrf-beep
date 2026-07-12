from PIL import Image
import mss

with mss.MSS() as sct:
    monitor = sct.monitors[1]  # primary monitor
    width, height = monitor["width"], monitor["height"]
    # crop bottom-right corner (approx 40% width, 35% height)
    left = int(width * 0.6)
    top = int(height * 0.65)
    right = width
    bottom = height
    img = sct.grab({"left": left, "top": top, "width": right - left, "height": bottom - top})
    pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
    pil_img.save("bottom_right.png")
    print(f"Saved bottom-right region: {right-left}x{bottom-top}")
