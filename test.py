import time
import mss
import numpy

def screen_record_efficient():
    # 800x600 windowed mode
    mon = {"top": 40, "left": 0, "width": 800, "height": 640}

    title = "[MSS] FPS benchmark"
    fps = 0
    sct = mss.mss()
    last_time = time.time()

    while time.time() - last_time < 1:
        img = numpy.asarray(sct.grab(mon))
        fps += 1

    return fps

print("MSS:", screen_record_efficient())