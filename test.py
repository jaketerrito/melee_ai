import time
import mss
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from percentages import get_percents

# Screenshot stress test
def screen_record_efficient():
    # 800x600 windowed mode
    mon = {"top": 0, "left": 0, "width": 320, "height": 240}

    title = "[MSS] FPS benchmark"
    fps = 0
    sct = mss.mss()
    last_time = time.time()

    while time.time() - last_time < 1:
        img = numpy.asarray(sct.grab(mon))
        fps += 1
    return fps

def main():
    sct = mss.mss()
    g = get_percents(sct, 1920, 1080, 0)
    while True:
        print(next(g))

main()
