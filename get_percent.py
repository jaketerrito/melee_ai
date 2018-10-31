import time
import mss
import numpy
from PIL import Image
import pyautogui as pg
import string
from random import *
min_char = 8
max_char = 12
allchar = string.ascii_letters

for i in range(0,301):
    mon = {"top": 886, "left": 521 - 66*2, "width": 66*3, "height": 76}
    sct = mss.mss()
    im = sct.grab(mon)
    im= Image.frombytes("RGB", im.size, im.bgra, "raw", "BGRX")
    rand = "".join(choice(allchar) for x in range(randint(min_char, max_char)))
    im.save(str(i) + '-' + rand + ".jpg","JPEG")
    pg.click(220, 255) 
    pg.click(220, 255)
    #emulation sped up 
    time.sleep(.25)
    

