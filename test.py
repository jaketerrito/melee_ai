import time
import mss
import numpy
import matplotlib.pyplot as plt

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

print("MSS:", screen_record_efficient())


# pip install pyautogui to get keyboard output library
# run this command to press a key:

import pyautogui as pg
pg.press("a")