#!/bin/bash
# FFW-SH5 MuJoCo Teleoperation launcher
cd "$(dirname "$0")"
PYTHONPATH=/home/ggh/.local/lib/python3.14/site-packages:/home/ggh/venv/lib/python3.14/site-packages:. \
    python3.14 main.py "$@"
