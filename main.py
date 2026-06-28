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


# ── Collision layer bit mask ──────────────────────────────────────────
# bit 1 (=1): 기본 레이어 — arm mesh, table, 캔
# bit 2 (=2): 손가락 캡슐 레이어 — 캡슐↔캡슐 충돌을 막고 캡슐↔캔만 허용
#   캡슐:  contype=2, conaffinity=0  (다른 캡슐/arm 안 봄)
#   캔:    conaffinity=3 (bit1+bit2) (캡슐도 봄)
_LAYER_NORMAL = 1
_LAYER_FINGER = 2


def _find_body(root, name: str):
    """MjSpec body tree 재귀 탐색."""
    if root.name == name:
        return root
    child = root.first_body()
    while child is not None:
        found = _find_body(child, name)
        if found:
            return found
        child = root.next_body(child)
    return None


def _collect_descendants(spec_body):
    """spec_body 직계 + 하위 모든 body 재귀 수집 (spec_body 자신 제외)."""
    out = []
    child = spec_body.first_body()
    while child is not None:
        out.append(child)
        out.extend(_collect_descendants(child))
        child = spec_body.next_body(child)
    return out


def _add_palm_fill(spec, base_name: str):
    """Palm body의 오목한 내부를 BOX geom으로 채워 캔 관통 방지.

    hx5_*_base의 로컬 Z축 = 월드 X+ (전방, 캔 방향).
    Palm 오목 내부(Z<0.054)를 채워 캔이 GJK 사각지대로 파고드는 것 방지.
    contype=1(normal layer) 로 설정해 캔(conaffinity=3)과 충돌한다.
    """
    body = _find_body(spec.worldbody, base_name)
    if body is None:
        return
    g = body.add_geom()
    g.type        = mujoco.mjtGeom.mjGEOM_BOX
    g.pos         = [0.001, 0.004, 0.030]
    g.size        = [0.023, 0.065, 0.033]  # Z: -0.003~+0.063, Y: ±6.9cm
    g.contype     = _LAYER_NORMAL
    g.conaffinity = _LAYER_NORMAL
    g.group       = 3
    g.density     = 0
    g.friction    = [2.0, 0.05, 0.01]
    g.solimp      = [0.95, 0.99, 0.001, 0.5, 2]
    g.rgba        = [0.2, 0.8, 0.2, 0.0]


def _replace_finger_mesh_collision(spec, model_ref):
    """손가락 mesh collision geom → capsule 교체.

    핵심: contype=2 / conaffinity=0 으로 캡슐끼리는 서로 충돌하지 않으면서
    캔(conaffinity=3)과만 충돌한다. 이전 시도에서 그립이 망가진 원인이
    바로 conaffinity=1로 설정된 캡슐들이 서로 밀어냈기 때문.

    mesh AABB 최장 축 방향으로 capsule을 정렬한다.
    """
    for palm_name in ('hx5_l_base', 'hx5_r_base'):
        palm_sb = _find_body(spec.worldbody, palm_name)
        if palm_sb is None:
            continue

        for sb in _collect_descendants(palm_sb):
            bname = sb.name
            if not bname:
                continue
            bid = mujoco.mj_name2id(model_ref, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                continue

            # ── 이 body의 mesh collision geom 찾기 ──────────────────
            coll_gid = -1
            for gi in range(model_ref.ngeom):
                if (int(model_ref.geom_bodyid[gi]) == bid
                        and model_ref.geom_contype[gi] == 1
                        and model_ref.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH):
                    coll_gid = gi
                    break
            if coll_gid < 0:
                continue

            # ── mesh vertex → body frame 변환 ────────────────────────
            gp    = model_ref.geom_pos[coll_gid]          # (3,) body-frame offset
            gq    = model_ref.geom_quat[coll_gid]         # (4,) [w,x,y,z]
            rot9  = np.zeros(9); mujoco.mju_quat2Mat(rot9, gq)
            R     = rot9.reshape(3, 3)
            mid   = int(model_ref.geom_dataid[coll_gid])
            va    = model_ref.mesh_vertadr[mid]
            nv    = model_ref.mesh_vertnum[mid]
            raw   = model_ref.mesh_vert[va:va+nv].reshape(-1, 3)
            verts = (R @ raw.T).T + gp                    # body-frame vertices

            xs, xe = float(verts[:,0].min()), float(verts[:,0].max())
            ys, ye = float(verts[:,1].min()), float(verts[:,1].max())
            zs, ze = float(verts[:,2].min()), float(verts[:,2].max())
            dx, dy, dz = xe-xs, ye-ys, ze-zs
            cx, cy, cz = (xs+xe)/2, (ys+ye)/2, (zs+ze)/2

            # ── 최장 축 → capsule 방향 결정 ──────────────────────────
            # MuJoCo capsule 기본 장축 = local Z.
            # 장축이 X면 90°@Y, Y면 -90°@X 회전.
            if dx >= dy and dx >= dz:
                r  = max(0.007, min(min(dy, dz) / 2 * 0.90, 0.016))
                hl = max(0.005, dx / 2 - r * 0.5)
                q  = [0.7071068, 0, 0.7071068, 0]   # Z→X: 90° around Y
            elif dy >= dz:
                r  = max(0.007, min(min(dx, dz) / 2 * 0.90, 0.016))
                hl = max(0.005, dy / 2 - r * 0.5)
                q  = [0.7071068, -0.7071068, 0, 0]  # Z→Y: -90° around X
            else:
                r  = max(0.007, min(min(dx, dy) / 2 * 0.90, 0.016))
                hl = max(0.005, dz / 2 - r * 0.5)
                q  = [1, 0, 0, 0]                   # Z: no rotation

            # ── 기존 mesh collision geom 비활성화 ────────────────────
            g_list = []
            g = sb.first_geom()
            while g is not None:
                if g.group == 3:
                    g_list.append(g)
                try:    g = sb.next_geom(g)
                except: break
            for g in g_list:
                g.contype = 0; g.conaffinity = 0

            # ── capsule 추가 ─────────────────────────────────────────
            cap             = sb.add_geom()
            cap.type        = mujoco.mjtGeom.mjGEOM_CAPSULE
            cap.pos         = [cx, cy, cz]
            cap.quat        = q
            cap.size        = [r, hl, 0]
            cap.contype     = _LAYER_NORMAL   # 1: standard layer
            cap.conaffinity = _LAYER_NORMAL   # 1: standard layer
            cap.group       = 3
            cap.density     = 0               # 질량 기여 없음
            cap.friction    = [2.0, 0.05, 0.01]
            cap.solimp      = [0.95, 0.99, 0.001, 0.5, 2]
            cap.margin      = 0.006           # 6mm early detection — gap 보완
            cap.rgba        = [0.3, 0.7, 1.0, 0.0]


def build_scene() -> mujoco.MjModel:
    """FFW-SH5 scene에 table + can을 동적으로 추가."""
    # ── 1차 컴파일: mesh AABB 측정용 (원본 XML, table/can 없음) ──────────
    _spec_ref  = mujoco.MjSpec.from_file(ORIG_SCENE)
    _model_ref = _spec_ref.compile()

    # ── 2차 컴파일: 실제 씬 구성 ─────────────────────────────────────────
    spec = mujoco.MjSpec.from_file(ORIG_SCENE)

    # 손가락 mesh collision → capsule 교체 (gap-free + self-collision-free)
    _replace_finger_mesh_collision(spec, _model_ref)

    # Palm body 오목 내부 채움 (concave mesh → GJK 사각지대 보완)
    _add_palm_fill(spec, 'hx5_l_base')
    _add_palm_fill(spec, 'hx5_r_base')

    wb = spec.worldbody

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

    # 캔 (dynamic, freejoint) — 테이블 위에 올려놓기
    CAN_HALF_H = 0.110             # 반높이 0.110m → 전체 높이 22cm
    CAN_Z = TABLE_H + CAN_HALF_H
    can = wb.add_body()
    can.name = 'can'
    can.pos  = [0.85, 0.15, CAN_Z]

    fj = can.add_freejoint()
    fj.name = 'can_free'

    cg = can.add_geom()
    cg.name     = 'can_geom'
    cg.type     = mujoco.mjtGeom.mjGEOM_CYLINDER
    cg.size     = [0.040, CAN_HALF_H, 0]
    cg.rgba     = [0.85, 0.15, 0.15, 1]
    cg.mass     = 0.20
    cg.friction = [2.0, 0.05, 0.01]

    model = spec.compile()

    # ── Post-compile: 캔 contact 파라미터 ────────────────────────────────
    _can_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'can_geom')
    if _can_gid >= 0:
        model.geom_condim[_can_gid]       = 4      # torsional friction (spin 방지)
        model.geom_solimp[_can_gid, 1]   = 0.99
        model.geom_margin[_can_gid]       = 0.001
        # conaffinity = bit1(normal) | bit2(finger capsules)
        # → 캔이 arm mesh(bit1)와 손가락 캡슐(bit2) 모두와 충돌
        model.geom_conaffinity[_can_gid]  = _LAYER_NORMAL

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
