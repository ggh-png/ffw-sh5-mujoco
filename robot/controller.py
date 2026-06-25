"""FFW-SH5 MuJoCo Teleoperation Controller — Full Feature Set.

Controls
--------
WASD      base translate (body-frame)
←/→       base yaw
Q/E       lift up/down
I/K       IK EE forward/back
J/L       IK EE lateral
U/O       IK EE up/down
hold 1    IK left arm only
hold 2    IK right arm only
Z/X       grip toggle (L/R)
F         camera-follow toggle
G         gizmo toggle
R         reset can to initial pose
F11       fullscreen toggle
"""
import math
import time

import numpy as np
import mujoco

from .keystate import KeyState
from .ik import dls_ik

try:
    import glfw as _glfw
    _HAS_GLFW = True
except ImportError:
    _HAS_GLFW = False

# ── GLFW key codes ──────────────────────────────────────────────────────
_K: dict[str, int] = {c: ord(c) for c in 'WASDQEIJKLUOZX12FGR'}
_K.update({'LEFT': 263, 'RIGHT': 262, 'F11': 300})

# ── Physical constants ──────────────────────────────────────────────────
BASE_MAX_SPD  = 0.55   # m/s
YAW_MAX_SPD   = 1.20   # rad/s
LIFT_STEP     = 0.005  # m per key-event (lift stays discrete)
IK_SPEED      = 0.40   # m/s continuous
WHEEL_RADIUS  = 0.090  # m
K_ACCEL       = 3.0    # m/s² linear acceleration
K_BRAKE       = 6.0    # m/s² linear braking
K_YAW_ACCEL   = 4.0    # rad/s² yaw acceleration
K_YAW_BRAKE   = 8.0    # rad/s² yaw braking
IK_WORKSPACE  = 0.78   # max reach from shoulder [m]
EMA_ALPHA     = 0.03   # loop-freq EMA smoothing

WHEEL_XY = {
    'left':  np.array([ 0.1371,  0.2554]),
    'right': np.array([ 0.1371, -0.2554]),
    'rear':  np.array([-0.2899,  0.0   ]),
}

ARM_L = [f'arm_l_joint{i}' for i in range(1, 8)]
ARM_R = [f'arm_r_joint{i}' for i in range(1, 8)]
FIN_L = [f'finger_l_joint{i}' for i in range(1, 21)]
FIN_R = [f'finger_r_joint{i}' for i in range(1, 21)]

OPEN_ANGLE: dict[str, float] = {}
for _jname in FIN_L:
    _idx = int(_jname.split('joint')[1])
    if _idx == 2:
        OPEN_ANGLE[_jname] = math.pi / 2
    elif _idx in (6, 10, 14, 18):
        OPEN_ANGLE[_jname] = math.pi / 2
    else:
        OPEN_ANGLE[_jname] = 0.0
for _jname in FIN_R:
    _idx = int(_jname.split('joint')[1])
    if _idx == 2:
        OPEN_ANGLE[_jname] = -math.pi / 2
    elif _idx in (6, 10, 14, 18):
        OPEN_ANGLE[_jname] = math.pi / 2
    else:
        OPEN_ANGLE[_jname] = 0.0


def _accel(cur: float, target: float, accel: float, brake: float, dt: float) -> float:
    """Linear acceleration / braking toward target."""
    if abs(target) > 1e-4:
        diff = target - cur
        step = accel * dt
        return cur + math.copysign(min(abs(diff), step), diff)
    else:
        step = brake * dt
        sign = math.copysign(1.0, cur)
        return sign * max(0.0, abs(cur) - step)


class TeleopController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m = model
        self.d = data
        self.ks = KeyState()

        def aid(name: str) -> int:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert i >= 0, f'actuator not found: {name}'
            return i

        def try_aid(name: str):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            return i if i >= 0 else None

        def jid(name: str) -> int:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert i >= 0, f'joint not found: {name}'
            return i

        def try_jid(name: str):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return i if i >= 0 else None

        def bid(name: str) -> int:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert i >= 0, f'body not found: {name}'
            return i

        # ── Actuator IDs ──────────────────────────────────────────────
        self._a_steer = {k: aid(f'{k}_wheel_steer') for k in WHEEL_XY}
        self._a_drive = {k: aid(f'{k}_wheel_drive') for k in WHEEL_XY}
        self._a_lift  = aid('lift_joint')
        self._a_arm_l = [aid(n) for n in ARM_L]
        self._a_arm_r = [aid(n) for n in ARM_R]
        self._a_fin_l = {n: try_aid(n) for n in FIN_L}
        self._a_fin_r = {n: try_aid(n) for n in FIN_R}

        # ── Joint addresses ────────────────────────────────────────────
        fj_id = jid('floating_base')
        self._fj_qpos = model.jnt_qposadr[fj_id]
        self._fj_dof  = model.jnt_dofadr[fj_id]

        lift_jid = jid('lift_joint')
        self._j_lift_qadr  = model.jnt_qposadr[lift_jid]
        self._j_lift_range = tuple(model.jnt_range[lift_jid])

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

        # ── EE body IDs ────────────────────────────────────────────────
        self._ee_l = bid('hx5_l_base')
        self._ee_r = bid('hx5_r_base')

        # ── Shoulder bodies (workspace clamping) ────────────────────────
        jl1_bid = model.jnt_bodyid[jid('arm_l_joint1')]
        jr1_bid = model.jnt_bodyid[jid('arm_r_joint1')]
        self._shoulder_l_bid = model.body_parentid[jl1_bid]
        self._shoulder_r_bid = model.body_parentid[jr1_bid]

        # ── Arm joint qpos ranges (for HUD display) ────────────────────
        self._arm_l_qadrs  = self._jl_qadrs
        self._arm_r_qadrs  = self._jr_qadrs
        self._arm_l_ranges = [tuple(model.jnt_range[jid(n)]) for n in ARM_L]
        self._arm_r_ranges = [tuple(model.jnt_range[jid(n)]) for n in ARM_R]

        # ── Can reset ──────────────────────────────────────────────────
        can_j = try_jid('can_free')
        if can_j is not None:
            self._can_qadr = model.jnt_qposadr[can_j]
            self._can_vadr = model.jnt_dofadr[can_j]
        else:
            self._can_qadr = self._can_vadr = None
        self._can_init_qpos: np.ndarray | None = None

        # ── Runtime state ──────────────────────────────────────────────
        self.base_yaw      = 0.0
        self._vx           = 0.0
        self._vy           = 0.0
        self._yaw_rate     = 0.0

        self._ik_tgt_l_base = np.zeros(3)
        self._ik_tgt_r_base = np.zeros(3)

        self._grip_l   = False
        self._grip_r   = False
        self._tgl_l_t  = -99.0
        self._tgl_r_t  = -99.0
        self._ik_err_l = 0.0
        self._ik_err_r = 0.0

        self.show_gizmo   = True
        self._cam_follow  = False
        self._fullscreen  = False

        self._wall_t0         = time.perf_counter()
        self._loop_freq_ema   = 60.0
        self._loop_dt         = 1.0 / 60.0

        # Public attributes read by markers.py
        self.ik_world_tgt_l = np.zeros(3)
        self.ik_world_tgt_r = np.zeros(3)
        self.ee_pos_l       = np.zeros(3)
        self.ee_pos_r       = np.zeros(3)
        self.base_world_pos = np.zeros(3)

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        qa = self._fj_qpos
        self.d.qpos[qa + 2] = 0.1465
        self.d.qpos[qa + 3] = 1.0
        mujoco.mj_forward(self.m, self.d)

        self.base_yaw  = 0.0
        self._vx = self._vy = self._yaw_rate = 0.0

        self._ik_tgt_l_base = self._world_to_base(self.d.xpos[self._ee_l].copy())
        self._ik_tgt_r_base = self._world_to_base(self.d.xpos[self._ee_r].copy())
        self._grip_l = self._grip_r = False

        if self._can_qadr is not None:
            self._can_init_qpos = self.d.qpos[self._can_qadr: self._can_qadr + 7].copy()

        self._apply_grip('l', 0.0)
        self._apply_grip('r', 0.0)
        self._wall_t0 = time.perf_counter()

    # ── Key callback (GLFW thread) ────────────────────────────────────────

    def on_key(self, key: int):
        self.ks.on_key(key)
        t = time.perf_counter()

        if key == _K['Z'] and (t - self._tgl_l_t) > 0.3:
            self._grip_l  = not self._grip_l
            self._tgl_l_t = t
        if key == _K['X'] and (t - self._tgl_r_t) > 0.3:
            self._grip_r  = not self._grip_r
            self._tgl_r_t = t

        if key == _K['F']:
            self._cam_follow = not self._cam_follow
        if key == _K['G']:
            self.show_gizmo = not self.show_gizmo
        if key == _K['R']:
            self._reset_can()
        if key == _K['F11']:
            self._toggle_fullscreen()

    # ── Main update ──────────────────────────────────────────────────────

    def update(self, dt: float):
        self._loop_dt = dt
        freq = 1.0 / dt if dt > 1e-6 else self._loop_freq_ema
        self._loop_freq_ema = (1.0 - EMA_ALPHA) * self._loop_freq_ema + EMA_ALPHA * freq

        self._update_base(dt)
        self._update_lift()
        self._update_ik_targets(dt)
        self._update_ik()
        self._update_grip()

    # ── Base movement (kinematic + wheel visuals) ─────────────────────────

    def _update_base(self, dt: float):
        ks = self.ks
        tvx  = (float(ks.is_down(_K['W'])) - float(ks.is_down(_K['S']))) * BASE_MAX_SPD
        tvy  = (float(ks.is_down(_K['A'])) - float(ks.is_down(_K['D']))) * BASE_MAX_SPD
        tyaw = (float(ks.is_down(_K['LEFT'])) - float(ks.is_down(_K['RIGHT']))) * YAW_MAX_SPD

        self._vx  = _accel(self._vx,  tvx,  K_ACCEL, K_BRAKE,     dt)
        self._vy  = _accel(self._vy,  tvy,  K_ACCEL, K_BRAKE,     dt)
        self._yaw_rate = _accel(self._yaw_rate, tyaw, K_YAW_ACCEL, K_YAW_BRAKE, dt)

        c  = math.cos(self.base_yaw)
        s_ = math.sin(self.base_yaw)
        wx = c * self._vx - s_ * self._vy
        wy = s_ * self._vx + c  * self._vy

        qa = self._fj_qpos
        self.d.qpos[qa + 0] += wx * dt
        self.d.qpos[qa + 1] += wy * dt
        self.base_yaw        += self._yaw_rate * dt

        hw = self.base_yaw / 2.0
        self.d.qpos[qa + 3] = math.cos(hw)
        self.d.qpos[qa + 4] = 0.0
        self.d.qpos[qa + 5] = 0.0
        self.d.qpos[qa + 6] = math.sin(hw)

        da = self._fj_dof
        self.d.qvel[da + 0] = wx
        self.d.qvel[da + 1] = wy
        self.d.qvel[da + 5] = self._yaw_rate

        for name, wxy in WHEEL_XY.items():
            wvx  = wx  - self._yaw_rate * wxy[1]
            wvy  = wy  + self._yaw_rate * wxy[0]
            spd  = math.sqrt(wvx ** 2 + wvy ** 2)
            ang  = math.atan2(wvy, wvx) if spd > 0.01 else 0.0
            sign = 1.0
            if ang > math.pi / 2:
                ang -= math.pi; sign = -1.0
            elif ang < -math.pi / 2:
                ang += math.pi; sign = -1.0
            self.d.ctrl[self._a_steer[name]] = ang
            self.d.ctrl[self._a_drive[name]] = sign * spd / WHEEL_RADIUS

        qa = self._fj_qpos
        self.base_world_pos = np.array([
            self.d.qpos[qa],
            self.d.qpos[qa + 1],
            self.d.qpos[qa + 2],
        ])

    # ── Lift ─────────────────────────────────────────────────────────────

    def _update_lift(self):
        qa      = self._j_lift_qadr
        lo, hi  = self._j_lift_range
        if self.ks.is_down(_K['Q']):
            self.d.qpos[qa] = float(np.clip(self.d.qpos[qa] + LIFT_STEP, lo, hi))
        if self.ks.is_down(_K['E']):
            self.d.qpos[qa] = float(np.clip(self.d.qpos[qa] - LIFT_STEP, lo, hi))
        self.d.ctrl[self._a_lift] = self.d.qpos[qa]

    # ── IK target update (velocity-based, in base frame) ─────────────────

    def _update_ik_targets(self, dt: float):
        ks   = self.ks
        do_l = not ks.is_down(_K['2'])
        do_r = not ks.is_down(_K['1'])

        fwd = float(ks.is_down(_K['I'])) - float(ks.is_down(_K['K']))
        lat = float(ks.is_down(_K['J'])) - float(ks.is_down(_K['L']))
        up  = float(ks.is_down(_K['U'])) - float(ks.is_down(_K['O']))

        delta = np.array([fwd, lat, up]) * (IK_SPEED * dt)

        if do_l:
            self._ik_tgt_l_base += delta
            self._ik_tgt_l_base  = self._clamp_workspace(
                self._ik_tgt_l_base, self._shoulder_l_bid)
        if do_r:
            self._ik_tgt_r_base += delta
            self._ik_tgt_r_base  = self._clamp_workspace(
                self._ik_tgt_r_base, self._shoulder_r_bid)

    # ── IK solve ─────────────────────────────────────────────────────────

    def _update_ik(self):
        mujoco.mj_forward(self.m, self.d)

        tgt_l = self._base_to_world(self._ik_tgt_l_base)
        tgt_r = self._base_to_world(self._ik_tgt_r_base)

        self.ik_world_tgt_l = tgt_l
        self.ik_world_tgt_r = tgt_r
        self.ee_pos_l = self.d.xpos[self._ee_l].copy()
        self.ee_pos_r = self.d.xpos[self._ee_r].copy()

        self._ik_err_l = dls_ik(
            self.m, self.d, self._ee_l, tgt_l,
            self._jl_dadrs, self._jl_qadrs)
        self._ik_err_r = dls_ik(
            self.m, self.d, self._ee_r, tgt_r,
            self._jr_dadrs, self._jr_qadrs)

        for aid, qadr in zip(self._a_arm_l, self._jl_qadrs):
            self.d.ctrl[aid] = self.d.qpos[qadr]
        for aid, qadr in zip(self._a_arm_r, self._jr_qadrs):
            self.d.ctrl[aid] = self.d.qpos[qadr]

    # ── Grip ─────────────────────────────────────────────────────────────

    def _update_grip(self):
        self._apply_grip('l', 1.0 if self._grip_l else 0.0)
        self._apply_grip('r', 1.0 if self._grip_r else 0.0)

    def _apply_grip(self, side: str, grip: float):
        s     = 1.0 if side == 'l' else -1.0
        fins  = FIN_L if side == 'l' else FIN_R
        info  = self._fin_l_info if side == 'l' else self._fin_r_info
        act   = self._a_fin_l    if side == 'l' else self._a_fin_r

        for jname in fins:
            if jname not in info:
                continue
            a_id = act.get(jname)
            if a_id is None:
                continue
            qadr, (lo, hi) = info[jname]
            idx = int(jname.split('joint')[1])
            open_val = OPEN_ANGLE.get(jname, 0.0)

            if idx <= 4:
                if idx == 2:
                    target = open_val
                elif idx in (3, 4):
                    close  = s * (-math.pi / 3)
                    target = open_val + (close - open_val) * grip
                else:
                    target = open_val
            else:
                phase = (idx - 5) % 4
                if phase == 0:
                    target = open_val - grip * 0.15
                elif phase == 1:
                    target = open_val + grip * (math.pi / 4)
                else:
                    target = open_val + grip * (math.pi / 3)

            self.d.ctrl[a_id] = float(np.clip(target, lo, hi))

    # ── Can reset ────────────────────────────────────────────────────────

    def _reset_can(self):
        if self._can_qadr is None or self._can_init_qpos is None:
            return
        self.d.qpos[self._can_qadr: self._can_qadr + 7] = self._can_init_qpos
        self.d.qvel[self._can_vadr: self._can_vadr + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)

    # ── Fullscreen (called from GLFW thread via on_key) ───────────────────

    def _toggle_fullscreen(self):
        if not _HAS_GLFW:
            return
        try:
            win = _glfw.get_current_context()
            if win is None:
                return
            if not self._fullscreen:
                mon  = _glfw.get_primary_monitor()
                mode = _glfw.get_video_mode(mon)
                _glfw.set_window_monitor(
                    win, mon, 0, 0,
                    mode.size.width, mode.size.height,
                    mode.refresh_rate)
                self._fullscreen = True
            else:
                _glfw.set_window_monitor(win, None, 80, 80, 1280, 720, 0)
                self._fullscreen = False
        except Exception:
            pass

    # ── Coordinate helpers ────────────────────────────────────────────────

    def _base_rot(self) -> np.ndarray:
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]])

    def _base_to_world(self, local: np.ndarray) -> np.ndarray:
        qa = self._fj_qpos
        base = np.array([self.d.qpos[qa], self.d.qpos[qa + 1], self.d.qpos[qa + 2]])
        return self._base_rot() @ local + base

    def _world_to_base(self, world: np.ndarray) -> np.ndarray:
        qa = self._fj_qpos
        base = np.array([self.d.qpos[qa], self.d.qpos[qa + 1], self.d.qpos[qa + 2]])
        return self._base_rot().T @ (world - base)

    def _clamp_workspace(self, tgt_base: np.ndarray, shoulder_bid: int) -> np.ndarray:
        shoulder_world = self.d.xpos[shoulder_bid].copy()
        shoulder_base  = self._world_to_base(shoulder_world)
        delta = tgt_base - shoulder_base
        dist  = np.linalg.norm(delta)
        if dist > IK_WORKSPACE:
            delta *= IK_WORKSPACE / dist
        return shoulder_base + delta

    # ── Overlay: HUD + camera tracking + 3D markers ──────────────────────

    def overlay(self, viewer):
        # Camera follow
        if self._cam_follow:
            try:
                qa = self._fj_qpos
                lx = float(self.d.qpos[qa])
                ly = float(self.d.qpos[qa + 1])
                lz = float(self.d.qpos[qa + 2]) + 0.5
                viewer.cam.lookat[0] = lx
                viewer.cam.lookat[1] = ly
                viewer.cam.lookat[2] = lz
            except Exception:
                pass

        # 3D markers
        try:
            from .markers import render as _render_markers
            _render_markers(viewer.user_scn, self)
        except Exception:
            pass

        # HUD text
        self._draw_hud(viewer)

    def _draw_hud(self, viewer):
        el  = self._ik_err_l * 1000.0
        er  = self._ik_err_r * 1000.0

        def col(e):
            return 'G' if e < 5.0 else ('Y' if e < 20.0 else 'R')

        qa   = self._fj_qpos
        bx   = float(self.d.qpos[qa])
        by   = float(self.d.qpos[qa + 1])
        lz   = float(self.d.qpos[self._j_lift_qadr])
        deg  = math.degrees(self.base_yaw)
        vspd = math.sqrt(self._vx ** 2 + self._vy ** 2)

        wall  = time.perf_counter() - self._wall_t0
        simtm = float(self.d.time)
        freq  = self._loop_freq_ema

        # IK target positions (base-frame for display)
        tl = self._ik_tgt_l_base
        tr = self._ik_tgt_r_base

        # Joint angles (degrees) for wrist display (joints 5,6,7)
        def jdeg(qadrs, i):
            return math.degrees(float(self.d.qpos[qadrs[i]]))

        wl5, wl6, wl7 = jdeg(self._arm_l_qadrs, 4), jdeg(self._arm_l_qadrs, 5), jdeg(self._arm_l_qadrs, 6)
        wr5, wr6, wr7 = jdeg(self._arm_r_qadrs, 4), jdeg(self._arm_r_qadrs, 5), jdeg(self._arm_r_qadrs, 6)

        def arm_angles(qadrs):
            return '  '.join(f'J{i+1}={math.degrees(float(self.d.qpos[qadrs[i]])):+6.1f}'
                              for i in range(7))

        topleft = (
            '=== Arm State ===\n'
            f'L: {arm_angles(self._arm_l_qadrs)}\n'
            f'   Wrist RPY  R={wl5:+6.1f}  P={wl6:+6.1f}  Y={wl7:+6.1f} deg\n'
            f'R: {arm_angles(self._arm_r_qadrs)}\n'
            f'   Wrist RPY  R={wr5:+6.1f}  P={wr6:+6.1f}  Y={wr7:+6.1f} deg\n'
            f'Lift: {lz:.3f} m'
        )

        topright = (
            '=== Telemetry ===\n'
            f'Sim:  {simtm:8.3f} s\n'
            f'Wall: {wall:8.1f} s\n'
            f'Freq: {freq:6.1f} Hz\n'
            f'IK L: {el:5.1f}mm [{col(el)}]   R: {er:5.1f}mm [{col(er)}]'
        )

        bottomleft = (
            f'Base  ({bx:+.2f},{by:+.2f}) yaw={deg:+.1f}d  |v|={vspd:.2f}m/s\n'
            f'Grip  L={"GRIP" if self._grip_l else "open"}  R={"GRIP" if self._grip_r else "open"}\n'
            f'TgtL  ({tl[0]:+.2f},{tl[1]:+.2f},{tl[2]:+.2f}) base-frame\n'
            f'TgtR  ({tr[0]:+.2f},{tr[1]:+.2f},{tr[2]:+.2f}) base-frame\n'
            f'Cam-follow={"ON" if self._cam_follow else "off"}  '
            f'Gizmo={"ON" if self.show_gizmo else "off"}  '
            f'Fullscr={"ON" if self._fullscreen else "off"}'
        )

        bottomright = (
            'WASD=move  ←/→=yaw  Q/E=lift\n'
            'IJKL UO=IK EE  1=L only  2=R only\n'
            'Z/X=grip   F=cam-follow  G=gizmo\n'
            'R=reset can   F11=fullscreen'
        )

        try:
            viewer.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_100,
                 mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 topleft, ''),
                (mujoco.mjtFontScale.mjFONTSCALE_100,
                 mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                 topright, ''),
                (mujoco.mjtFontScale.mjFONTSCALE_150,
                 mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 bottomleft, ''),
                (mujoco.mjtFontScale.mjFONTSCALE_100,
                 mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                 '', bottomright),
            ])
        except Exception:
            try:
                viewer.set_texts([(
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    bottomleft,
                    bottomright,
                )])
            except Exception:
                pass
