# melee_ai
A bot that will beat you in Super Smash Brothers Brawl

How to setup Dolphin for pip input on Linux (locations may change for mac, see readme in https://github.com/levthedev/gamecube_IO)
1. See https://wiki.dolphin-emu.org/index.php?title=Installing_Dolphin#Linux to install dolphin
2. Navigate to ~/.config/dolphin-emu
3. mkdir -p Profiles/GCPad
4. create a new file called pipe.ini with these contents:

[GCPad1]
Device = Pipe/0/pipe1
Buttons/A = `Button A`
Buttons/B = `Button B`
Buttons/X = `Button X`
Buttons/Y = `Button Y`
Buttons/Z = `Button Z`
Buttons/Start = `Button START`
D-Pad/Up = `Button D_UP`
D-Pad/Down = `Button D_DOWN`
D-Pad/Left = `Button D_LEFT`
D-Pad/Right = `Button D_RIGHT`
Triggers/L = `Button L`
Triggers/R = `Button R`
Main Stick/Up = `Axis MAIN Y -`
Main Stick/Down =  `Axis MAIN Y +`
Main Stick/Left = `Axis MAIN X -`
Main Stick/Right = `Axis MAIN X +`
C-Stick/Up = `Axis C Y -`
C-Stick/Down =  `Axis C Y +`
C-Stick/Left = `Axis C X -`
C-Stick/Right = `Axis C X +`


5. Navigate to ~/.local/share/dolphin-emu
6. mkdir Pipes
7. mkfifo Pipes/pipe
8. chmod 777 Pipes/pipe
9. In Dolphin, go to Controllers settings
10. Set port one to 'Standard Controller', open controller config
11. Load 'pipe' profile, set device to Pipe/0/pipe
12. Enable 'Background Input'
13. hit OK, run game

Only use the original Melee 1.02 NTCS rom, found here: https://the-eye.eu/public/rom/Nintendo%20Gamecube/
under Super Smash Bros Melee.7z
