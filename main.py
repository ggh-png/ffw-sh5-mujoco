#!/usr/bin/env python3
"""
FFW-SH5 MuJoCo Teleoperation
=================================
Run:  cd /home/ggh/ffw-sh5-mujoco && bash run.sh

Controls
--------
↑/↓             베이스 전후 이동
←/→             베이스 yaw 회전
Q/E             리프트 상승/하강
Tab             FK ↔ IK 모드 전환

IK 모드
  I/K J/L U/O  EE 이동 (전후, 좌우, 상하)
  9             자세 IK 토글  (ON시: 3/4=roll 5/6=pitch 7/8=yaw)
  0             손바닥 중심 IK 토글 (캔 파지 후 기울이기 — 손바닥 중심 기준 회전)
  hold 1        왼팔만
  hold 2        오른팔만

FK 모드
  1/2           왼팔 / 오른팔 선택
  [/]           조인트 선택 (J1 ↔ J7)
  I/K           선택 조인트 증가/감소
  Home/End/Del  최대/최소/영점

Hand (홀드)
  Z/C           왼손 3지(검지·중지·약지) 닫기/열기
  X/V           오른손 3지 닫기/열기
  A/S           왼손 엄지 닫기/열기
  H/N           오른손 엄지 닫기/열기
  (새끼손가락 항상 편 상태 / 오른손은 태스크 중 TASK 표시)

P               캔 따르기 태스크 시작/중단
F               카메라 추적 토글
G               기즈모 토글
R               캔 초기 위치로 리셋
F11             전체화면 토글
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
from robot.gui import ControlPanel
from robot.task import CanPourTask

# ── 렌더 / 물리 비율 설정 ──────────────────────────────────────────────
# physics timestep 은 model.opt.timestep (보통 0.002s = 500 Hz)
# 렌더는 ~60 Hz → 1프레임당 약 8 물리 스텝
RENDER_HZ  = 60
N_SUBSTEPS = 8   # 물리 스텝 수 per 렌더 프레임 (= 500/60 ≈ 8)


def build_scene() -> mujoco.MjModel:
    """FFW-SH5 scene에 table + can을 동적으로 추가."""
    spec = mujoco.MjSpec.from_file(ORIG_SCENE)
    wb   = spec.worldbody

    # 테이블 (static)
    table = wb.add_body()
    table.name = 'table'
    table.pos  = [0.8, 0, 0]

    TABLE_H = 0.65   # 테이블 상면 높이 (m)
    top = table.add_geom()
    top.name = 'table_top'
    top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.size = [0.30, 0.35, 0.02]
    top.pos  = [0, 0, TABLE_H - 0.02]
    top.rgba = [0.60, 0.40, 0.20, 1]

    leg_half = (TABLE_H - 0.04) / 2
    for i, (lx, ly) in enumerate([(-0.27, -0.32), (-0.27,  0.32),
                                    ( 0.27, -0.32), ( 0.27,  0.32)]):
        leg = table.add_geom()
        leg.name = f'table_leg{i+1}'
        leg.type = mujoco.mjtGeom.mjGEOM_BOX
        leg.size = [0.02, 0.02, leg_half]
        leg.pos  = [lx, ly, leg_half]
        leg.rgba = [0.50, 0.30, 0.15, 1]

    # 캔 (dynamic, freejoint) — 테이블 위에 올려놓기 (can half-height = 0.055m)
    CAN_Z = TABLE_H + 0.055
    can = wb.add_body()
    can.name = 'can'
    can.pos  = [0.85, 0.15, CAN_Z]

    fj = can.add_freejoint()
    fj.name = 'can_free'

    cg = can.add_geom()
    cg.name     = 'can_geom'
    cg.type     = mujoco.mjtGeom.mjGEOM_CYLINDER
    cg.size     = [0.033, 0.055, 0]
    cg.rgba     = [0.85, 0.15, 0.15, 1]
    cg.mass     = 0.20          # lighter can — easier to lift (was 0.35 kg)
    cg.friction = [2.0, 0.05, 0.01]  # rubber-grip surface (was 0.8, 0.005, 0.0001)

    model = spec.compile()

    # Post-compile contact tuning for can
    _can_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'can_geom')
    if _can_gid >= 0:
        model.geom_condim[_can_gid]    = 4     # prevents can spinning in grasp
        model.geom_solimp[_can_gid, 1] = 0.99  # stiffer (was 0.95)
        model.geom_margin[_can_gid]    = 0.001 # 1mm early contact detection

    return model


def main():
    model = build_scene()
    data  = mujoco.MjData(model)

    ctrl = TeleopController(model, data)
    task = CanPourTask(model, data)
    ctrl.task = task

    with mujoco.viewer.launch_passive(
        model, data,
        key_callback=ctrl.on_key,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        ctrl.reset()
        # GUI must start AFTER MuJoCo viewer has claimed its GL/EGL context.
        # Starting it before causes an EGL context conflict ("could not create window").
        gui = ControlPanel(ctrl)

        frame_dt = 1.0 / RENDER_HZ

        while viewer.is_running():
            t0 = time.perf_counter()

            # ── 자율 태스크 스텝 (렌더 프레임마다 1회) ───────────────
            task.step(ctrl)

            # ── 물리 서브스텝 ─────────────────────────────────────────
            # IK 는 마지막 스텝에만 실행 (비용 절감)
            sub_dt = model.opt.timestep
            for step in range(N_SUBSTEPS):
                run_ik = (step == N_SUBSTEPS - 1)
                ctrl.update(sub_dt, run_ik=run_ik)
                mujoco.mj_step(model, data)
                ctrl.apply_floor_constraint()
                ctrl.apply_spread_lock()

            # ── 렌더 ─────────────────────────────────────────────────
            ctrl.overlay(viewer)
            viewer.sync()

            # ── 프레임 타이밍 ─────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            sleep_t = frame_dt - elapsed
            if sleep_t > 0.001:
                time.sleep(sleep_t)


if __name__ == '__main__':
    main()
