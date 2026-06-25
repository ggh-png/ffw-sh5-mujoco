"""FFW-SH5 MuJoCo Teleoperation Controller.

Base movement
-------------
UP / DOWN     Forward / backward (body-frame)
LEFT / RIGHT  Yaw left / right
Q / E         Lift up / down

IK mode (default)
-----------------
I / K         EE forward / backward
J / L         EE lateral left / right
U / O         EE up / down
hold 1        Left arm only
hold 2        Right arm only

FK mode
-------
Tab           Toggle FK / IK
1 / 2         Select left / right arm
[ / ]         Cycle joint J1..J7
I / K         Adjust selected joint angle

Common
------
Z / X         Left / right grip toggle
F             Camera-follow toggle
G             Gizmo toggle
R             Reset can
F11           Fullscreen toggle
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

# ── GLFW key codes (all ASCII-range keys) ──────────────────────────────
_K: dict[str, int] = {c: ord(c) for c in 'QEIJKLUOZX12FGR'}
_K.update({
    'UP':     265,   # forward
    'DOWN':   264,   # backward
    'LEFT':   263,   # yaw left
    'RIGHT':  262,   # yaw right
    'F11':    300,
    'TAB':    258,
    'LBRACK': 91,    # [
    'RBRACK': 93,    # ]
    'HOME':   268,   # FK: joint → max
    'END':    269,   # FK: joint → min
    'DEL':    261,   # FK: joint → 0
})

# ── Physical constants ──────────────────────────────────────────────────
BASE_MAX_SPD  = 0.55   # m/s
YAW_MAX_SPD   = 1.20   # rad/s
IK_SPEED      = 0.40   # m/s
FK_SPEED      = 0.80   # rad/s
LIFT_STEP     = 0.003  # m per update
WHEEL_RADIUS  = 0.090  # m
K_ACCEL       = 3.0
K_BRAKE       = 6.0
K_YAW_ACCEL   = 4.0
K_YAW_BRAKE   = 8.0
IK_WORKSPACE  = 0.78   # m from shoulder
EMA_ALPHA     = 0.05

# IK target display ranges (base-frame, meters)
IK_X_RANGE = (-0.50, 1.50)
IK_Y_RANGE = (-1.00, 1.00)
IK_Z_RANGE = ( 0.00, 2.00)

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
for _jn in FIN_L:
    _ix = int(_jn.split('joint')[1])
    OPEN_ANGLE[_jn] = (math.pi / 2 if _ix == 2 else
                       math.pi / 2 if _ix in (6, 10, 14, 18) else 0.0)
for _jn in FIN_R:
    _ix = int(_jn.split('joint')[1])
    OPEN_ANGLE[_jn] = (-math.pi / 2 if _ix == 2 else
                        math.pi / 2 if _ix in (6, 10, 14, 18) else 0.0)


# ── ASCII progress bar helpers ──────────────────────────────────────────

def _pbar(val: float, lo: float, hi: float, w: int = 18) -> str:
    """Return an ASCII progress bar of total length w+3.

    '#' = cursor position
    '|' = zero/centre marker (only when range spans zero)
    '=' = filled region between zero and cursor
    '-' = empty
    trailing char: '*' if near limit, ' ' otherwise
    """
    if hi <= lo:
        return '[' + '-' * w + '] '
    pct = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    pos = int(round(pct * (w - 1)))

    if lo < 0.0 < hi:
        mid = int(round((-lo) / (hi - lo) * (w - 1)))
    else:
        mid = 0
    mid = max(0, min(w - 1, mid))

    bar = ['-'] * w
    if pos >= mid:
        for i in range(mid, pos + 1):
            bar[i] = '='
    else:
        for i in range(pos, mid + 1):
            bar[i] = '='
    bar[mid] = '|'
    bar[pos] = '#'

    near = '*' if (pct < 0.05 or pct > 0.95) else ' '
    return '[' + ''.join(bar) + ']' + near


def _ik_err_tag(err_mm: float) -> str:
    if err_mm < 5.0:
        return 'OK'
    if err_mm < 20.0:
        return '~'
    return '!'


def _accel(cur: float, tgt: float, ac: float, br: float, dt: float) -> float:
    if abs(tgt) > 1e-4:
        diff = tgt - cur
        return cur + math.copysign(min(abs(diff), ac * dt), diff)
    step = br * dt
    return math.copysign(max(0.0, abs(cur) - step), cur) if cur != 0.0 else 0.0


# ── Controller ──────────────────────────────────────────────────────────

class TeleopController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m = model
        self.d = data
        self.ks = KeyState()

        def aid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert i >= 0, f'actuator not found: {name}'
            return i

        def try_aid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            return i if i >= 0 else None

        def jid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert i >= 0, f'joint not found: {name}'
            return i

        def try_jid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return i if i >= 0 else None

        def bid(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert i >= 0, f'body not found: {name}'
            return i

        # Actuator IDs
        self._a_steer = {k: aid(f'{k}_wheel_steer') for k in WHEEL_XY}
        self._a_drive = {k: aid(f'{k}_wheel_drive') for k in WHEEL_XY}
        self._a_lift  = aid('lift_joint')
        self._a_arm_l = [aid(n) for n in ARM_L]
        self._a_arm_r = [aid(n) for n in ARM_R]
        self._a_fin_l = {n: try_aid(n) for n in FIN_L}
        self._a_fin_r = {n: try_aid(n) for n in FIN_R}

        # Joint addresses
        fj = jid('floating_base')
        self._fj_qpos = model.jnt_qposadr[fj]
        self._fj_dof  = model.jnt_dofadr[fj]

        lft_j = jid('lift_joint')
        self._j_lift_qadr  = model.jnt_qposadr[lft_j]
        self._j_lift_range = tuple(model.jnt_range[lft_j])

        def arm_addrs(names):
            dadrs, qadrs, ranges = [], [], []
            for n in names:
                j = jid(n)
                dadrs.append(model.jnt_dofadr[j])
                qadrs.append(model.jnt_qposadr[j])
                ranges.append(tuple(model.jnt_range[j]))
            return dadrs, qadrs, ranges

        self._jl_dadrs, self._jl_qadrs, self._jl_ranges = arm_addrs(ARM_L)
        self._jr_dadrs, self._jr_qadrs, self._jr_ranges = arm_addrs(ARM_R)

        def fin_info(names):
            res = {}
            for n in names:
                i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                if i >= 0:
                    res[n] = (model.jnt_qposadr[i], tuple(model.jnt_range[i]))
            return res

        self._fin_l_info = fin_info(FIN_L)
        self._fin_r_info = fin_info(FIN_R)

        # EE / shoulder bodies
        self._ee_l = bid('hx5_l_base')
        self._ee_r = bid('hx5_r_base')

        jl1_bid = model.jnt_bodyid[jid('arm_l_joint1')]
        jr1_bid = model.jnt_bodyid[jid('arm_r_joint1')]
        self._shoulder_l_bid = model.body_parentid[jl1_bid]
        self._shoulder_r_bid = model.body_parentid[jr1_bid]

        # Can reset
        can_j = try_jid('can_free')
        if can_j is not None:
            self._can_qadr = model.jnt_qposadr[can_j]
            self._can_vadr = model.jnt_dofadr[can_j]
        else:
            self._can_qadr = self._can_vadr = None
        self._can_init_qpos: np.ndarray | None = None

        # Base velocity / pose state (kinematic model)
        self._vx       = 0.0
        self._vy       = 0.0
        self._yaw_rate = 0.0
        self._yaw_des  = 0.0     # integrated desired yaw angle
        self._base_z   = 0.1465  # fixed floor height (no gravity sinking)
        self._win      = None    # GLFW window handle (cached from key callback)

        # IK state
        self._ik_tgt_l_base = np.zeros(3)
        self._ik_tgt_r_base = np.zeros(3)
        self._ik_err_l = 0.0
        self._ik_err_r = 0.0

        # FK state
        self._mode     = 'ik'
        self._fk_joint = 0
        self._fk_arm   = 'l'

        # Grip state (0.0-1.0)
        self._grip_l  = 0.0
        self._grip_r  = 0.0
        self._tgl_l_t = -99.0
        self._tgl_r_t = -99.0

        # Misc state
        self.show_gizmo  = True
        self._cam_follow = False
        self._fullscreen = False
        self._wall_t0    = time.perf_counter()
        self._freq_ema   = 60.0

        # Public cache (read by markers.py)
        self.ik_world_tgt_l = np.zeros(3)
        self.ik_world_tgt_r = np.zeros(3)
        self.ee_pos_l       = np.zeros(3)
        self.ee_pos_r       = np.zeros(3)
        self.base_world_pos = np.zeros(3)

    # ── base_yaw — uses integrated kinematic state ───────────────────────

    @property
    def base_yaw(self) -> float:
        return self._yaw_des

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        qa = self._fj_qpos
        self.d.qpos[qa + 2] = 0.1465
        self.d.qpos[qa + 3] = 1.0
        mujoco.mj_forward(self.m, self.d)

        self._vx = self._vy = self._yaw_rate = 0.0
        self._yaw_des = 0.0
        self._base_z  = float(self.d.qpos[qa + 2])
        self._ik_tgt_l_base = self._world_to_base(self.d.xpos[self._ee_l].copy())
        self._ik_tgt_r_base = self._world_to_base(self.d.xpos[self._ee_r].copy())
        self._grip_l = self._grip_r = 0.0
        self._mode = 'ik'

        if self._can_qadr is not None:
            self._can_init_qpos = self.d.qpos[self._can_qadr: self._can_qadr + 7].copy()

        self._apply_grip('l', 0.0)
        self._apply_grip('r', 0.0)
        self._wall_t0 = time.perf_counter()

        for k in WHEEL_XY:
            self.d.ctrl[self._a_steer[k]] = 0.0
            self.d.ctrl[self._a_drive[k]] = 0.0

    # ── Key callback (GLFW thread) ────────────────────────────────────────

    def on_key(self, key: int):
        # Cache GLFW window handle the first time we're called (we're in the
        # GLFW thread here so get_current_context() is reliable).
        if _HAS_GLFW and self._win is None:
            try:
                w = _glfw.get_current_context()
                if w and w != 0:
                    self._win = w
            except Exception:
                pass

        self.ks.on_key(key)
        t = time.perf_counter()

        # Grip toggle (full open ↔ full close)
        if key == _K['Z'] and (t - self._tgl_l_t) > 0.3:
            self._grip_l  = 0.0 if self._grip_l > 0.5 else 1.0
            self._tgl_l_t = t
        if key == _K['X'] and (t - self._tgl_r_t) > 0.3:
            self._grip_r  = 0.0 if self._grip_r > 0.5 else 1.0
            self._tgl_r_t = t

        # Mode switch
        if key == _K['TAB']:
            if self._mode == 'ik':
                self._mode = 'fk'
            else:
                self._ik_tgt_l_base = self._world_to_base(
                    self.d.xpos[self._ee_l].copy())
                self._ik_tgt_r_base = self._world_to_base(
                    self.d.xpos[self._ee_r].copy())
                self._mode = 'ik'

        # FK joint selection + quick-set
        if self._mode == 'fk':
            if key == _K['LBRACK']:
                self._fk_joint = (self._fk_joint - 1) % 7
            if key == _K['RBRACK']:
                self._fk_joint = (self._fk_joint + 1) % 7
            if key == _K['1']:
                self._fk_arm = 'l'
            if key == _K['2']:
                self._fk_arm = 'r'
            # Home/End/Del: jump selected joint to limit or zero
            if key in (_K['HOME'], _K['END'], _K['DEL']):
                self._fk_jump(key)

        if key == _K['F']:
            self._cam_follow = not self._cam_follow
        if key == _K['G']:
            self.show_gizmo = not self.show_gizmo
        if key == _K['R']:
            self._reset_can()
        if key == _K['F11']:
            self._toggle_fullscreen()

    # ── Main update ──────────────────────────────────────────────────────

    def update(self, dt: float, run_ik: bool = True):
        freq = 1.0 / dt if dt > 1e-6 else self._freq_ema
        self._freq_ema = (1.0 - EMA_ALPHA) * self._freq_ema + EMA_ALPHA * freq

        self._update_base(dt)
        self._update_lift(dt)

        if self._mode == 'ik':
            self._update_ik_targets(dt)
            if run_ik:
                self._update_ik()
        else:
            self._update_fk(dt)

        self._update_grip()

    # ── Base (kinematic pose + visual wheel animation) ────────────────────

    def _update_base(self, dt: float):
        ks = self.ks
        tvx  = (float(ks.is_down(_K['UP']))   - float(ks.is_down(_K['DOWN'])))  * BASE_MAX_SPD
        tyaw = (float(ks.is_down(_K['LEFT'])) - float(ks.is_down(_K['RIGHT']))) * YAW_MAX_SPD

        self._vx       = _accel(self._vx,       tvx,  K_ACCEL,     K_BRAKE,     dt)
        self._yaw_rate = _accel(self._yaw_rate, tyaw,  K_YAW_ACCEL, K_YAW_BRAKE, dt)

        # Integrate desired yaw
        self._yaw_des += self._yaw_rate * dt

        c, s_ = math.cos(self._yaw_des), math.sin(self._yaw_des)
        wx = c * self._vx
        wy = s_ * self._vx

        # KINEMATIC: directly set base qpos — prevents gravity sinking
        qa = self._fj_qpos
        self.d.qpos[qa + 0] += wx * dt
        self.d.qpos[qa + 1] += wy * dt
        self.d.qpos[qa + 2]  = self._base_z   # locked height

        hw = self._yaw_des * 0.5
        self.d.qpos[qa + 3] = math.cos(hw)
        self.d.qpos[qa + 4] = 0.0
        self.d.qpos[qa + 5] = 0.0
        self.d.qpos[qa + 6] = math.sin(hw)

        # Set qvel so Jacobian-based IK sees correct base velocity
        da = self._fj_dof
        self.d.qvel[da + 0] = wx
        self.d.qvel[da + 1] = wy
        self.d.qvel[da + 2] = 0.0
        self.d.qvel[da + 3] = 0.0
        self.d.qvel[da + 4] = 0.0
        self.d.qvel[da + 5] = self._yaw_rate

        # Visual wheel animation (steer + drive actuators)
        for name, wxy in WHEEL_XY.items():
            wvx = wx  - self._yaw_rate * wxy[1]
            wvy = wy  + self._yaw_rate * wxy[0]
            spd = math.sqrt(wvx ** 2 + wvy ** 2)
            ang = math.atan2(wvy, wvx) if spd > 0.01 else 0.0
            sign = 1.0
            if ang > math.pi / 2:
                ang -= math.pi; sign = -1.0
            elif ang < -math.pi / 2:
                ang += math.pi; sign = -1.0
            self.d.ctrl[self._a_steer[name]] = ang
            self.d.ctrl[self._a_drive[name]] = sign * spd / WHEEL_RADIUS

        self.base_world_pos = np.array([float(self.d.qpos[qa]),
                                        float(self.d.qpos[qa + 1]),
                                        float(self.d.qpos[qa + 2])])

    # ── Lift ─────────────────────────────────────────────────────────────

    def _update_lift(self, dt: float):
        qa     = self._j_lift_qadr
        lo, hi = self._j_lift_range
        delta  = (float(self.ks.is_down(_K['Q'])) -
                  float(self.ks.is_down(_K['E']))) * LIFT_STEP
        if delta != 0.0:
            self.d.qpos[qa] = float(np.clip(self.d.qpos[qa] + delta, lo, hi))
        self.d.ctrl[self._a_lift] = self.d.qpos[qa]

    # ── IK target update (base-frame, continuous velocity) ───────────────

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
            self._ik_tgt_l_base = self._clamp_ws(self._ik_tgt_l_base,
                                                   self._shoulder_l_bid)
        if do_r:
            self._ik_tgt_r_base += delta
            self._ik_tgt_r_base = self._clamp_ws(self._ik_tgt_r_base,
                                                   self._shoulder_r_bid)

    # ── IK solver ────────────────────────────────────────────────────────

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

    # ── FK quick-set (Home/End/Del) ───────────────────────────────────────

    def _fk_jump(self, key: int):
        """Jump selected joint to max (Home), min (End), or zero (Del)."""
        arm    = self._fk_arm
        qadrs  = self._jl_qadrs if arm == 'l' else self._jr_qadrs
        aids   = self._a_arm_l  if arm == 'l' else self._a_arm_r
        ranges = self._jl_ranges if arm == 'l' else self._jr_ranges
        j      = self._fk_joint
        lo, hi = ranges[j]
        if key == _K['HOME']:
            val = hi
        elif key == _K['END']:
            val = lo
        else:  # DEL
            val = float(np.clip(0.0, lo, hi))
        self.d.qpos[qadrs[j]] = val
        self.d.ctrl[aids[j]]  = val
        mujoco.mj_forward(self.m, self.d)

    # ── FK direct joint control ───────────────────────────────────────────

    def _update_fk(self, dt: float):
        self.ee_pos_l = self.d.xpos[self._ee_l].copy()
        self.ee_pos_r = self.d.xpos[self._ee_r].copy()
        self.ik_world_tgt_l = self.ee_pos_l
        self.ik_world_tgt_r = self.ee_pos_r

        delta = (float(self.ks.is_down(_K['I'])) -
                 float(self.ks.is_down(_K['K']))) * FK_SPEED * dt
        if abs(delta) < 1e-9:
            return

        arm    = self._fk_arm
        qadrs  = self._jl_qadrs if arm == 'l' else self._jr_qadrs
        aids   = self._a_arm_l  if arm == 'l' else self._a_arm_r
        ranges = self._jl_ranges if arm == 'l' else self._jr_ranges

        j      = self._fk_joint
        qadr   = qadrs[j]
        lo, hi = ranges[j]
        new_q  = float(np.clip(float(self.d.qpos[qadr]) + delta, lo, hi))
        self.d.qpos[qadr]    = new_q
        self.d.ctrl[aids[j]] = new_q

        mujoco.mj_forward(self.m, self.d)
        self.ee_pos_l = self.d.xpos[self._ee_l].copy()
        self.ee_pos_r = self.d.xpos[self._ee_r].copy()
        self.ik_world_tgt_l = self.ee_pos_l
        self.ik_world_tgt_r = self.ee_pos_r

    # ── Grip ─────────────────────────────────────────────────────────────

    def _update_grip(self):
        self._apply_grip('l', self._grip_l)
        self._apply_grip('r', self._grip_r)

    def _apply_grip(self, side: str, grip: float):
        s    = 1.0 if side == 'l' else -1.0
        fins = FIN_L if side == 'l' else FIN_R
        info = self._fin_l_info if side == 'l' else self._fin_r_info
        act  = self._a_fin_l    if side == 'l' else self._a_fin_r

        for jname in fins:
            if jname not in info:
                continue
            a_id = act.get(jname)
            if a_id is None:
                continue
            qadr, (lo, hi) = info[jname]
            idx      = int(jname.split('joint')[1])
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

    # ── Fullscreen / window resize ───────────────────────────────────────

    def _toggle_fullscreen(self):
        """F11: toggle fullscreen using the cached GLFW window handle."""
        if not _HAS_GLFW or self._win is None:
            return
        try:
            if not self._fullscreen:
                mon  = _glfw.get_primary_monitor()
                mode = _glfw.get_video_mode(mon)
                _glfw.set_window_monitor(
                    self._win, mon, 0, 0,
                    mode.size.width, mode.size.height, mode.refresh_rate)
                self._fullscreen = True
            else:
                # Restore to a windowed size; position at top-left with margin
                _glfw.set_window_monitor(
                    self._win, None, 80, 80, 1280, 720, 0)
                self._fullscreen = False
        except Exception as e:
            print(f'[ctrl] fullscreen error: {e}')

    # ── Coordinate helpers ────────────────────────────────────────────────

    def _rot(self) -> np.ndarray:
        yaw = self.base_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _base_to_world(self, local: np.ndarray) -> np.ndarray:
        qa = self._fj_qpos
        bp = np.array([float(self.d.qpos[qa]),
                        float(self.d.qpos[qa + 1]),
                        float(self.d.qpos[qa + 2])])
        return self._rot() @ local + bp

    def _world_to_base(self, world: np.ndarray) -> np.ndarray:
        qa = self._fj_qpos
        bp = np.array([float(self.d.qpos[qa]),
                        float(self.d.qpos[qa + 1]),
                        float(self.d.qpos[qa + 2])])
        return self._rot().T @ (world - bp)

    def _clamp_ws(self, tgt_base: np.ndarray, shoulder_bid: int) -> np.ndarray:
        sh_world = self.d.xpos[shoulder_bid].copy()
        sh_base  = self._world_to_base(sh_world)
        delta    = tgt_base - sh_base
        dist     = np.linalg.norm(delta)
        if dist > IK_WORKSPACE:
            delta *= IK_WORKSPACE / dist
        return sh_base + delta

    # ── Overlay entry point ───────────────────────────────────────────────

    def overlay(self, viewer):
        if self._cam_follow:
            try:
                qa = self._fj_qpos
                viewer.cam.lookat[0] = float(self.d.qpos[qa])
                viewer.cam.lookat[1] = float(self.d.qpos[qa + 1])
                viewer.cam.lookat[2] = float(self.d.qpos[qa + 2]) + 0.5
            except Exception:
                pass

        try:
            from .markers import render as _rm
            _rm(viewer.user_scn, self)
        except Exception:
            pass

        self._draw_hud(viewer)

    # ── HUD drawing ──────────────────────────────────────────────────────

    def _draw_hud(self, viewer):
        topleft   = self._panel_joints()
        topright  = self._panel_ik_and_telemetry()
        bottomleft  = self._panel_status()
        bottomright = self._panel_controls()

        texts = [
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
        ]
        try:
            viewer.set_texts(texts)
        except Exception:
            try:
                viewer.set_texts([texts[2]])
            except Exception:
                pass

    # ── Panel: joint angles ───────────────────────────────────────────────

    def _panel_joints(self) -> str:
        PW = 14  # bar width

        def arm_block(label, qadrs, ranges, sel_j, active):
            lines = [f'--- {label} ---']
            for i in range(7):
                val     = float(self.d.qpos[qadrs[i]])
                lo, hi  = ranges[i]
                deg     = math.degrees(val)
                lo_d    = math.degrees(lo)
                hi_d    = math.degrees(hi)
                bar     = _pbar(val, lo, hi, PW)
                sel     = '>' if (active and i == sel_j) else ' '
                limit   = '!' if (bar.endswith('*')) else ' '
                lines.append(
                    f'{sel}J{i+1} {bar} {deg:+7.1f}  [{lo_d:+.0f}~{hi_d:+.0f}]{limit}')
            return '\n'.join(lines)

        fk_j   = self._fk_joint if self._mode == 'fk' else -1
        fk_arm = self._fk_arm   if self._mode == 'fk' else ''

        lz    = float(self.d.qpos[self._j_lift_qadr])
        lo, hi = self._j_lift_range
        lift_bar = _pbar(lz, lo, hi, PW)

        return (
            arm_block('Left Arm',  self._jl_qadrs, self._jl_ranges,
                      fk_j, fk_arm == 'l') +
            '\n\n' +
            arm_block('Right Arm', self._jr_qadrs, self._jr_ranges,
                      fk_j, fk_arm == 'r') +
            f'\n\n--- Lift ---\n {lift_bar} {lz*1000:+.0f}mm'
        )

    # ── Panel: IK targets + telemetry + hand ─────────────────────────────

    def _panel_ik_and_telemetry(self) -> str:
        PW = 16
        tl  = self._ik_tgt_l_base
        tr  = self._ik_tgt_r_base
        el  = self._ik_err_l * 1000.0
        er  = self._ik_err_r * 1000.0
        tag_l = _ik_err_tag(el)
        tag_r = _ik_err_tag(er)

        sim_t  = float(self.d.time)
        wall_t = time.perf_counter() - self._wall_t0
        freq   = self._freq_ema

        mode_str = self._mode.upper()
        arm_str  = ('BOTH' if self._mode == 'ik' else
                    ('L' if self._fk_arm == 'l' else 'R'))

        # IK target bars - show L/R on consecutive lines per axis
        def ik_rows():
            axes   = [('X(fwd)', tl[0], tr[0], *IK_X_RANGE),
                      ('Y(lat)', tl[1], tr[1], *IK_Y_RANGE),
                      ('Z( up)', tl[2], tr[2], *IK_Z_RANGE)]
            rows = []
            for name, lv, rv, lo, hi in axes:
                lb = _pbar(lv, lo, hi, PW)
                rb = _pbar(rv, lo, hi, PW)
                rows.append(f' L {name} {lb} {lv:+.3f}m')
                rows.append(f' R {name} {rb} {rv:+.3f}m')
                rows.append('')
            return '\n'.join(rows)

        # Wrist (joints 5,6,7) as compact bars
        def wrist_row(label, qadrs, ranges):
            parts = []
            for i in range(4, 7):
                v      = float(self.d.qpos[qadrs[i]])
                lo, hi = ranges[i]
                bar    = _pbar(v, lo, hi, 8)
                deg    = math.degrees(v)
                parts.append(f'J{i+1}{bar}{deg:+.0f}')
            return f' {label}: ' + '  '.join(parts)

        # Grip bars
        gl_pct = int(self._grip_l * 100)
        gr_pct = int(self._grip_r * 100)
        # 0% grip is the normal open state; suppress near-limit warning
        gl_bar = _pbar(self._grip_l, -0.01, 1.0, 12)[:-1] + ' '
        gr_bar = _pbar(self._grip_r, -0.01, 1.0, 12)[:-1] + ' '
        gl_str = 'GRIP' if self._grip_l > 0.5 else 'open'
        gr_str = 'GRIP' if self._grip_r > 0.5 else 'open'

        return '\n'.join([
            '--- Telemetry ---',
            f' Sim  {sim_t:8.3f}s   Wall {wall_t:7.1f}s',
            f' Freq {freq:7.1f} Hz',
            f' IK-L {el:6.1f}mm [{tag_l}]   IK-R {er:6.1f}mm [{tag_r}]',
            '',
            f'--- IK Targets  Mode:{mode_str} Arm:{arm_str} ---',
            ik_rows(),
            '--- Wrist (J5/6/7) ---',
            wrist_row('L', self._jl_qadrs, self._jl_ranges),
            wrist_row('R', self._jr_qadrs, self._jr_ranges),
            '',
            '--- Hand ---',
            f' L ({gl_str}) Z={gl_bar} {gl_pct:3d}%',
            f' R ({gr_str}) X={gr_bar} {gr_pct:3d}%',
        ])

    # ── Panel: status ─────────────────────────────────────────────────────

    def _panel_status(self) -> str:
        qa  = self._fj_qpos
        bx  = float(self.d.qpos[qa])
        by  = float(self.d.qpos[qa + 1])
        deg = math.degrees(self.base_yaw)
        spd = abs(self._vx)
        yr  = math.degrees(self._yaw_rate)

        mode_disp = self._mode.upper()
        if self._mode == 'fk':
            j      = self._fk_joint
            qadrs  = self._jl_qadrs if self._fk_arm == 'l' else self._jr_qadrs
            ranges = self._jl_ranges if self._fk_arm == 'l' else self._jr_ranges
            cur_deg = math.degrees(float(self.d.qpos[qadrs[j]]))
            lo_deg  = math.degrees(ranges[j][0])
            hi_deg  = math.degrees(ranges[j][1])
            mode_disp += (f'  [{self._fk_arm.upper()}] J{j+1}'
                          f'  {cur_deg:+.1f}deg  [{lo_deg:+.0f}~{hi_deg:+.0f}]')

        cam_s = 'ON' if self._cam_follow else 'off'
        giz_s = 'ON' if self.show_gizmo  else 'off'
        fs_s  = 'ON' if self._fullscreen  else 'off'

        win_hint = ('F11=exit-FS' if self._fullscreen
                    else 'drag-edge=resize  drag-title=move  F11=fullscreen')

        return '\n'.join([
            f'Mode:{mode_disp}',
            f'Base ({bx:+.2f},{by:+.2f}) yaw={deg:+.1f}  spd={spd:.2f}m/s  rot={yr:+.1f}deg/s',
            f'Cam:{cam_s} Gizmo:{giz_s} FS:{fs_s}   {win_hint}',
        ])

    # ── Panel: controls ───────────────────────────────────────────────────

    def _panel_controls(self) -> str:
        fk_sel = (f'[{self._fk_arm.upper()}] J{self._fk_joint+1}'
                  if self._mode == 'fk' else '---')
        return '\n'.join([
            '--- Controls ---',
            'UP/DN=fwd/bk   LT/RT=yaw',
            'Q/E=lift up/dn',
            'Tab=FK/IK mode',
            '',
            '[ IK mode ]',
            ' I/K J/L U/O = EE fwd/bk lat up/dn',
            ' hold 1=L-only  hold 2=R-only',
            '',
            '[ FK mode ]  sel:' + fk_sel,
            ' 1/2=arm  [/]=joint  I/K=angle',
            ' Home=max  End=min  Del=zero',
            '',
            'Z/X=grip toggle',
            'F=cam-follow  G=gizmo  R=reset-can',
            'F11=fullscreen',
        ])
