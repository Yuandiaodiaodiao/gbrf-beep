import mss

with mss.mss() as sct:
    print("Monitors:", sct.monitors)
    filename = sct.shot(mon=-1, output="screenshot.png")
    print("Saved:", filename)
