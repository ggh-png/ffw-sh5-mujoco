#!/usr/bin/env python3
"""
FFW-SH5 MuJoCo Teleoperation
=================================
Run:  cd /home/ggh/ffw-sh5-mujoco && bash run.sh

Controls
--------
WASD         base translate (body-frame)
←/→          base yaw
Q/E          lift up/down
I/K J/L U/O  IK EE move (fwd/bk, lateral, up/dn)
hold 1       IK left arm only
hold 2       IK right arm only
Z/X          left/right grip toggle
F            camera-follow toggle
G            gizmo toggle
R            reset can to initial pose
F11          fullscreen toggle
"""
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

ASSET_BASE = os.path.join(SCRIPT_DIR, 'assets', 'ffw_sh5', 'robotis_ffw')
ORIG_SCENE = os.path.join(ASSET_BASE, 'scene_ffw_sh5.xml')

from robot.controller import TeleopController


def build_scene() -> mujoco.MjModel:
    """원본 FFW-SH5 scene에 table + can을 추가하여 MjModel 반환."""
    spec = mujoco.MjSpec.from_file(ORIG_SCENE)
    wb   = spec.worldbody

    # ── 테이블 (static) ────────────────────────────────────────────────
    table      = wb.add_body()
    table.name = 'table'
    table.pos  = [0.8, 0, 0]

    top       = table.add_geom()
    top.name  = 'table_top'
    top.type  = mujoco.mjtGeom.mjGEOM_BOX
    top.size  = [0.30, 0.35, 0.02]
    top.pos   = [0, 0, 0.42]
    top.rgba  = [0.60, 0.40, 0.20, 1]

    for i, (lx, ly) in enumerate([(-0.27, -0.32), (-0.27, 0.32),
                                    ( 0.27, -0.32), ( 0.27, 0.32)]):
        leg      = table.add_geom()
        leg.name = f'table_leg{i+1}'
        leg.type = mujoco.mjtGeom.mjGEOM_BOX
        leg.size = [0.02, 0.02, 0.21]
        leg.pos  = [lx, ly, 0.21]
        leg.rgba = [0.50, 0.30, 0.15, 1]

    # ── 캔 (dynamic, freejoint) ────────────────────────────────────────
    can      = wb.add_body()
    can.name = 'can'
    can.pos  = [0.80, 0, 0.50]

    fj      = can.add_freejoint()
    fj.name = 'can_free'

    cg          = can.add_geom()
    cg.name     = 'can_geom'
    cg.type     = mujoco.mjtGeom.mjGEOM_CYLINDER
    cg.size     = [0.033, 0.055, 0]
    cg.rgba     = [0.85, 0.15, 0.15, 1]
    cg.mass     = 0.35
    cg.friction = [0.8, 0.005, 0.0001]

    return spec.compile()


def main():
    model = build_scene()
    data  = mujoco.MjData(model)

    ctrl = TeleopController(model, data)

    with mujoco.viewer.launch_passive(
        model, data,
        key_callback=ctrl.on_key,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        ctrl.reset()

        prev_t = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            dt  = min(now - prev_t, 0.05)
            prev_t = now

            ctrl.update(dt)
            mujoco.mj_step(model, data)

            ctrl.overlay(viewer)
            viewer.sync()

            sleep_t = model.opt.timestep - (time.perf_counter() - now)
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == '__main__':
    main()
