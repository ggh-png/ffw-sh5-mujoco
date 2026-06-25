#!/bin/bash
# FFW-SH5 MuJoCo Teleoperation launcher
cd "$(dirname "$0")"

# Force X11 backend so GLFW gets title bar + resize handles (libdecor-gtk
# fails on this system; X11/XWayland always provides window decorations).
export XDG_SESSION_TYPE=x11
export GDK_BACKEND=x11

PYTHONPATH=/home/ggh/.local/lib/python3.14/site-packages:/home/ggh/venv/lib/python3.14/site-packages:. \
    python3.14 main.py "$@"
