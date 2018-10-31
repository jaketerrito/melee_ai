import time
import mss
import numpy as np
import sys
import os


def p_and_r(button):
    pcmd = "echo \"PRESS " + button + "\" > ~/.local/share/dolphin-emu/Pipes/pipe"
    rcmd = "echo \"RELEASE " + button + "\" > ~/.local/share/dolphin-emu/Pipes/pipe"
    exec_action(pcmd, rcmd, 60)

def set_stick(stick, x, y):
    scmd = "echo \"SET " + stick + " " + str(x) + " " + str(y) + "\" > ~/.local/share/dolphin-emu/Pipes/pipe"
    os.system(scmd)    

def reset_stick(stick):
    uscmd = "echo \"SET " + stick + " 0.5 0.5\" > ~/.local/share/dolphin-emu/Pipes/pipe"
    os.system(uscmd)

def exec_action(startcmd, endcmd, lim):
    os.system(startcmd)
    time.sleep(1/lim)
    os.system(endcmd)

def start_game_from_char_select():
    button_arr = ["A", "B", "Z", "L", "R", "Y", "X"]

    # Move to and Select Falco
    set_stick("MAIN", 0.5, 0)
    time.sleep(24/60)   
    reset_stick("MAIN")
    p_and_r("A")

    # Move to and enable CPU opponent
    set_stick("MAIN", 0.5, 1)
    time.sleep(8/60)
    set_stick("MAIN", 1, 0.5)
    time.sleep(11/60)
    reset_stick("MAIN")
    p_and_r("A")
    set_stick("MAIN", 0.5, 1)
    time.sleep(10/60)
    reset_stick("MAIN")
    p_and_r("A")
    set_stick("MAIN", 1, 0.5)
    time.sleep(8/60)
    reset_stick("MAIN")
    p_and_r("A")

    # Enter match select screen
    time.sleep(1/4)
    p_and_r("START")    

    # Move to and start a game on Battlefield
    time.sleep(1)
    set_stick("MAIN", 0.5, 0)
    time.sleep(2/60)
    reset_stick("MAIN")
    p_and_r("A")

    time.sleep(3)

    start_time = time.time()
    while(time.time() - start_time < 120):
        ri = np.random.randint(0, 7)
        p_and_r(button_arr[ri])
        rmone = np.random.randint(0, 4)
        rmtwo = np.random.randint(0, 4)
        set_stick("MAIN", rmone / 4, rmtwo / 4)   
        time.sleep(1/60)

start_game_from_char_select()
