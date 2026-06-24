"""FFW-SH5 MuJoCo Teleoperation Controller.

기능:
  - WASD / 방향키: 베이스 이동 (kinematic freejoint)
  - Q / E        : 리프트 상승 / 하강
  - I/K J/L U/O  : 양팔 EE IK 이동 (1/2 키로 좌/우 단독)
  - Z / X        : 좌/우 손 파지 토글
"""
import time
import math

import numpy as np
import mujoco

from .keystate import KeyState
from .ik       import dls_ik

# ── GLFW key codes ────────────────────────────────────────────────────
K = {c: ord(c) for c in 'WASDQEIJKLUOZX12'}
K['LEFT']  = 263
K['RIGHT'] = 262

# ── 상수 ──────────────────────────────────────────────────────────────
BASE_SPD      = 0.50   # m/s
YAW_SPD       = 0.80   # rad/s
LIFT_STEP     = 0.005  # m per key event
IK_STEP       = 0.005  # m per key event
WHEEL_RADIUS  = 0.090  # m

WHEEL_XY = {                              # (x, y) in base frame
    'left':  np.array([ 0.1371,  0.2554]),
    'right': np.array([ 0.1371, -0.2554]),
    'rear':  np.array([-0.2899,  0.0   ]),
}

ARM_L = [f'arm_l_joint{i}' for i in range(1, 8)]
ARM_R = [f'arm_r_joint{i}' for i in range(1, 8)]
FIN_L = [f'finger_l_joint{i}' for i in range(1, 21)]
FIN_R = [f'finger_r_joint{i}' for i in range(1, 21)]

# 기본 open 각도 (MJCF 기준: 좌 엄지 MCPyaw=+90°, 우=-90°, 4손가락 PIP=90°)
OPEN_ANGLE: dict[str, float] = {}
for jname in FIN_L:
    idx = int(jname.split('joint')[1])
    if idx == 2:                     # 좌 엄지 MCPyaw
        OPEN_ANGLE[jname] = math.pi / 2
    elif idx in (6, 10, 14, 18):     # 4손가락 PIP
        OPEN_ANGLE[jname] = math.pi / 2
    else:
        OPEN_ANGLE[jname] = 0.0
for jname in FIN_R:
    idx = int(jname.split('joint')[1])
    if idx == 2:                     # 우 엄지 MCPyaw
        OPEN_ANGLE[jname] = -math.pi / 2
    elif idx in (6, 10, 14, 18):
        OPEN_ANGLE[jname] = math.pi / 2
    else:
        OPEN_ANGLE[jname] = 0.0


class TeleopController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m = model
        self.d = data
        self.ks = KeyState()

        # ── actuator ID 캐싱 ─────────────────────────────────────────
        def aid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert i >= 0, f'actuator not found: {name}'
            return i

        self._a_steer = {k: aid(f'{k}_wheel_steer') for k in WHEEL_XY}
        self._a_drive = {k: aid(f'{k}_wheel_drive') for k in WHEEL_XY}
        self._a_lift  = aid('lift_joint')
        self._a_arm_l = [aid(n) for n in ARM_L]
        self._a_arm_r = [aid(n) for n in ARM_R]

        def try_aid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            return i if i >= 0 else None

        self._a_fin_l = {n: try_aid(n) for n in FIN_L}
        self._a_fin_r = {n: try_aid(n) for n in FIN_R}

        # ── joint 주소 캐싱 ───────────────────────────────────────────
        def jid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert i >= 0, f'joint not found: {name}'
            return i

        fj = jid('floating_base')
        self._fj_qpos = model.jnt_qposadr[fj]
        self._fj_dof  = model.jnt_dofadr[fj]

        self._j_lift_qadr = model.jnt_qposadr[jid('lift_joint')]
        self._j_lift_range = tuple(model.jnt_range[jid('lift_joint')])

        def arm_addrs(names):
            dadrs, qadrs = [], []
            for n in names:
                j = jid(n)
                dadrs.append(model.jnt_dofadr[j])
                qadrs.append(model.jnt_qposadr[j])
            return dadrs, qadrs

        self._jl_dadrs, self._jl_qadrs = arm_addrs(ARM_L)
        self._jr_dadrs, self._jr_qadrs = arm_addrs(ARM_R)

        def fin_info(names):
            res = {}
            for n in names:
                i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                if i >= 0:
                    res[n] = (model.jnt_qposadr[i], tuple(model.jnt_range[i]))
            return res

        self._fin_l_info = fin_info(FIN_L)
        self._fin_r_info = fin_info(FIN_R)

        # ── EE body IDs ───────────────────────────────────────────────
        def bid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert i >= 0, f'body not found: {name}'
            return i

        self._ee_l = bid('hx5_l_base')
        self._ee_r = bid('hx5_r_base')

        # ── 런타임 상태 ───────────────────────────────────────────────
        self._base_yaw  = 0.0
        self._ik_tgt_l  = np.zeros(3)
        self._ik_tgt_r  = np.zeros(3)
        self._grip_l    = False
        self._grip_r    = False
        self._tgl_l_t   = -99.0
        self._tgl_r_t   = -99.0
        self._ik_err_l  = 0.0
        self._ik_err_r  = 0.0

    # ── 리셋 ──────────────────────────────────────────────────────────

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        qa = self._fj_qpos
        self.d.qpos[qa + 2] = 0.1465   # z: 바퀴가 바닥에 닿도록
        self.d.qpos[qa + 3] = 1.0      # quaternion w=1 (no rotation)
        mujoco.mj_forward(self.m, self.d)

        self._ik_tgt_l = self.d.xpos[self._ee_l].copy()
        self._ik_tgt_r = self.d.xpos[self._ee_r].copy()
        self._base_yaw = 0.0
        self._grip_l = self._grip_r = False

        self._apply_grip('l', 0.0)
        self._apply_grip('r', 0.0)

    # ── 키 이벤트 ─────────────────────────────────────────────────────

    def on_key(self, key: int):
        self.ks.on_key(key)
        t = time.perf_counter()

        # 파지 토글 (cooldown 0.3 s)
        if key == K['Z'] and (t - self._tgl_l_t) > 0.3:
            self._grip_l  = not self._grip_l
            self._tgl_l_t = t
        if key == K['X'] and (t - self._tgl_r_t) > 0.3:
            self._grip_r  = not self._grip_r
            self._tgl_r_t = t

        # IK 타겟 delta (PRESS+REPEAT으로 연속 호출 → 연속 이동)
        s  = IK_STEP
        do_l = not self.ks.is_down(K['2'])   # 2 안 누르면 좌팔 포함
        do_r = not self.ks.is_down(K['1'])   # 1 안 누르면 우팔 포함

        if key == K['I']:
            if do_l: self._ik_tgt_l[0] += s
            if do_r: self._ik_tgt_r[0] += s
        if key == K['K']:
            if do_l: self._ik_tgt_l[0] -= s
            if do_r: self._ik_tgt_r[0] -= s
        if key == K['J']:
            if do_l: self._ik_tgt_l[1] += s
            if do_r: self._ik_tgt_r[1] += s
        if key == K['L']:
            if do_l: self._ik_tgt_l[1] -= s
            if do_r: self._ik_tgt_r[1] -= s
        if key == K['U']:
            if do_l: self._ik_tgt_l[2] += s
            if do_r: self._ik_tgt_r[2] += s
        if key == K['O']:
            if do_l: self._ik_tgt_l[2] -= s
            if do_r: self._ik_tgt_r[2] -= s

    # ── 메인 업데이트 ─────────────────────────────────────────────────

    def update(self, dt: float):
        self._update_base(dt)
        self._update_lift()
        self._update_ik()
        self._update_grip()

    # ── 베이스 이동 (kinematic) ───────────────────────────────────────

    def _update_base(self, dt: float):
        ks = self.ks
        vx  = (ks.is_down(K['W'])     ) * BASE_SPD \
            - (ks.is_down(K['S'])     ) * BASE_SPD
        vy  = (ks.is_down(K['A'])     ) * BASE_SPD \
            - (ks.is_down(K['D'])     ) * BASE_SPD
        yaw = (ks.is_down(K['LEFT'])  ) * YAW_SPD  \
            - (ks.is_down(K['RIGHT']) ) * YAW_SPD

        c  = math.cos(self._base_yaw)
        s_ = math.sin(self._base_yaw)
        wx = c * vx - s_ * vy
        wy = s_ * vx + c  * vy

        qa = self._fj_qpos
        self.d.qpos[qa + 0] += wx  * dt
        self.d.qpos[qa + 1] += wy  * dt
        self._base_yaw       += yaw * dt

        hw = self._base_yaw / 2
        self.d.qpos[qa + 3] = math.cos(hw)
        self.d.qpos[qa + 4] = 0.0
        self.d.qpos[qa + 5] = 0.0
        self.d.qpos[qa + 6] = math.sin(hw)

        da = self._fj_dof
        self.d.qvel[da + 0] = wx
        self.d.qvel[da + 1] = wy
        self.d.qvel[da + 5] = yaw

        # 바퀴 actuator (시각적 회전 + steering)
        for name, wxy in WHEEL_XY.items():
            wvx  = wx  - yaw * wxy[1]
            wvy  = wy  + yaw * wxy[0]
            spd  = math.sqrt(wvx**2 + wvy**2)
            ang  = math.atan2(wvy, wvx) if spd > 0.01 else 0.0
            sign = 1.0
            if ang > math.pi / 2:
                ang -= math.pi; sign = -1.0
            elif ang < -math.pi / 2:
                ang += math.pi; sign = -1.0
            self.d.ctrl[self._a_steer[name]] = ang
            self.d.ctrl[self._a_drive[name]] = sign * spd / WHEEL_RADIUS

    # ── 리프트 ────────────────────────────────────────────────────────

    def _update_lift(self):
        qa    = self._j_lift_qadr
        lo, hi = self._j_lift_range
        if self.ks.is_down(K['Q']):
            self.d.qpos[qa] = float(np.clip(self.d.qpos[qa] + LIFT_STEP, lo, hi))
        if self.ks.is_down(K['E']):
            self.d.qpos[qa] = float(np.clip(self.d.qpos[qa] - LIFT_STEP, lo, hi))
        self.d.ctrl[self._a_lift] = self.d.qpos[qa]

    # ── IK ────────────────────────────────────────────────────────────

    def _update_ik(self):
        mujoco.mj_forward(self.m, self.d)
        self._ik_err_l = dls_ik(
            self.m, self.d, self._ee_l, self._ik_tgt_l,
            self._jl_dadrs, self._jl_qadrs,
        )
        self._ik_err_r = dls_ik(
            self.m, self.d, self._ee_r, self._ik_tgt_r,
            self._jr_dadrs, self._jr_qadrs,
        )
        for aid, qadr in zip(self._a_arm_l, self._jl_qadrs):
            self.d.ctrl[aid] = self.d.qpos[qadr]
        for aid, qadr in zip(self._a_arm_r, self._jr_qadrs):
            self.d.ctrl[aid] = self.d.qpos[qadr]

    # ── 파지 ─────────────────────────────────────────────────────────

    def _update_grip(self):
        self._apply_grip('l', 1.0 if self._grip_l else 0.0)
        self._apply_grip('r', 1.0 if self._grip_r else 0.0)

    def _apply_grip(self, side: str, grip: float):
        s      = 1.0 if side == 'l' else -1.0
        fins   = FIN_L if side == 'l' else FIN_R
        info   = self._fin_l_info if side == 'l' else self._fin_r_info
        act    = self._a_fin_l    if side == 'l' else self._a_fin_r

        for jname in fins:
            if jname not in info:
                continue
            a_id = act.get(jname)
            if a_id is None:
                continue
            qadr, (lo, hi) = info[jname]
            idx = int(jname.split('joint')[1])   # 1-20

            open_val = OPEN_ANGLE.get(jname, 0.0)

            if idx <= 4:   # ── 엄지 ──────────────────────────────────
                if idx == 2:                     # MCPyaw: 항상 고정
                    target = open_val
                elif idx in (3, 4):              # MCPpitch / IP
                    close = s * (-math.pi / 3)
                    target = open_val + (close - open_val) * grip
                else:                            # CMC (idx=1)
                    target = open_val
            else:          # ── 4손가락 ───────────────────────────────
                phase = (idx - 5) % 4            # 0=MCP-spr, 1=PIP, 2=DIP, 3=TIP
                if phase == 0:                   # MCP-spread: 닫힐 때 조금 오므림
                    target = open_val - grip * 0.15
                elif phase == 1:                 # PIP: 90° base + 45° close
                    target = open_val + grip * (math.pi / 4)
                else:                            # DIP, TIP
                    target = open_val + grip * (math.pi / 3)

            self.d.ctrl[a_id] = float(np.clip(target, lo, hi))

    # ── 텍스트 오버레이 ───────────────────────────────────────────────

    def overlay(self, viewer):
        el = self._ik_err_l * 1000
        er = self._ik_err_r * 1000

        def col(e): return 'G' if e < 5 else ('Y' if e < 20 else 'R')

        qa = self._fj_qpos
        bx, by = self.d.qpos[qa], self.d.qpos[qa + 1]
        lz  = self.d.qpos[self._j_lift_qadr]
        deg = math.degrees(self._base_yaw)

        left_col = (
            f'IK err  L={el:5.1f}mm[{col(el)}] R={er:5.1f}mm[{col(er)}]\n'
            f'Base     ({bx:.2f},{by:.2f}) yaw={deg:.1f}d  lift={lz:.3f}m\n'
            f'Grip     L={"GRIP" if self._grip_l else "open"}  '
            f'R={"GRIP" if self._grip_r else "open"}'
        )
        right_col = (
            'WASD=move  ←/→=yaw  Q/E=lift up/down\n'
            'I/K=EE fwd/bk  J/L=lat  U/O=up/down\n'
            '1=L only  2=R only  Z/X=grip toggle'
        )
        try:
            viewer.set_texts([(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                left_col,
                right_col,
            )])
        except Exception:
            pass
