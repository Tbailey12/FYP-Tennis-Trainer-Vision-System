Final Year Engineering Project - Tennis Trainer Vision System

Two Raspberry Pi 4s are required, with a PiCamera V2 connected to each. Both Pis should be connected to a LAN or WLAN. The 'client.py' script runs on each Raspberry Pi and the 'server.py' script runs on another computer connected to the network, with the server IP address stored in 'consts.py'.

## Requires
Python 3.8 

OpenCV 4.1.1

## Main Scripts
*Most up to date code is in /code_refactor*

**server.py** - Controls both PiCamera V2s, waits for clients to start and connect

**client.py** - Sends/Receives commands from server, maintains gaussian background image while idle, detects moving objects and sends object details to server while recording

**ode_solve.py** - Solves a set of differential equations for tennis ball motion using initial conditions
