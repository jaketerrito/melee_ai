import time
import numpy as np
import sys
import os


# takes a button and presses then releases it
def p_and_r(button):
    press(button)
    release(button)

# presses a button
def press(button):
    pcmd = "echo \"PRESS " + button + "\" > ~/.local/share/dolphin-emu/Pipes/pipe"
    exec_action(pcmd, 60)

# releases a button
def release(button):
    rcmd = "echo \"RELEASE " + button + "\" > ~/.local/share/dolphin-emu/Pipes/pipe"
    exec_action(rcmd, 60)

# sets a stick to (x, y)
# (0, 0) is the top left direction
# (1, 1) is the bottom right
def set_stick(stick, x, y):
    scmd = "echo \"SET " + stick + " " + str(x) + " " + str(y) + "\" > ~/.local/share/dolphin-emu/Pipes/pipe"
    os.system(scmd)    

# sets stick back to (0.5, 0.5) the center position
def reset_stick(stick):
    uscmd = "echo \"SET " + stick + " 0.5 0.5\" > ~/.local/share/dolphin-emu/Pipes/pipe"
    os.system(uscmd)

# execs a start action (like PRESS) then waits a frame
def exec_action(cmd, lim):
    os.system(cmd)
    time.sleep(1/lim)

# Cleans up the pipe
def make_clean(button_arr):
    reset_stick("MAIN")
    for but in button_arr:
        release(but)

def start_game_from_char_select():
    button_arr = ["A", "B", "Z", "L"]

    # Clean up the pipe so no previous inputs are in it
    # Only necessary if a script crashed for some reason
    make_clean(button_arr)
    

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
    time.sleep(9/60)
    reset_stick("MAIN")
    p_and_r("A")

    # Enter match select screen
    time.sleep(1/2)
    p_and_r("START")    

    # Move to and start a game on Battlefield
    time.sleep(1)
    set_stick("MAIN", 0.5, 0)
    time.sleep(2/60)
    reset_stick("MAIN")
    p_and_r("A")

    time.sleep(3)

    start_time = time.time()
    frames = 0
    ri = 0
    while(time.time() - start_time < 10):
        rmone = np.random.uniform()
        rmtwo = np.random.uniform()
        set_stick("MAIN", rmone, rmtwo)
        if(frames % 2 == 1):
            ri = np.random.randint(0, len(button_arr))
            press(button_arr[ri])
        else:
            release(button_arr[ri])
        frames = frames + 1

    # Clean the pipe by reseting the stick and release everything
    make_clean(button_arr)

def do_rand_actions():
    button_arr = ["A", "B", "Z", "L"]
    frames = 0
    ri = 0
    start_time = time.time()
    while(time.time() - start_time < 120):
        rmone = np.random.uniform()
        rmtwo = np.random.uniform()
        set_stick("MAIN", rmone, rmtwo)
        if(frames % 2 == 1):
            ri = np.random.randint(0, len(button_arr))
            press(button_arr[ri])
        else:
            release(button_arr[ri])
        frames = frames + 1

    # Clean the pipe by reseting the stick and release everything
    make_clean(button_arr)

# Process on training:
# Time battle, time on 99 min
# We select characters etc. We start the script in game right at start of match
# Keras model with custom loss

do_rand_actions()
#start_game_from_char_select()
