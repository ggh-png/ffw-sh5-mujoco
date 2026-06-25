"""FFW-SH5 MuJoCo Teleoperation Controller.

Mode
----
Tab         FK / IK 모드 전환

IK 모드 (기본)
--------------
WASD        베이스 이동 (로봇 기준계)
←/→         베이스 yaw 회전
Q/E         리프트 상승/하강
I/K J/L U/O IK EE 이동 (전/후, 좌/우, 상/하)
hold 1      왼팔만
hold 2      오른팔만
Z/X         좌/우 그립 토글

FK 모드
-------
1/2         왼팔 / 오른팔 선택
[/]         관절 선택 (J1 ↔ J7)
I/K         선택 조인트 증가/감소

공통
----
F           카메라 추적 토글
G           기즈모 토글
R           캔 리셋
F11         전체화면 토글
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
_K.update({
    'LEFT':    263,
    'RIGHT':   262,
    'F11':     300,
    'TAB':     258,
    'LBRACK':  91,    # [
    'RBRACK':  93,    # ]
})

# ── 상수 ───────────────────────────────────────────────────────────────
BASE_MAX_SPD  = 0.55   # m/s
YAW_MAX_SPD   = 1.20   # rad/s
LIFT_STEP     = 0.003  # m per frame (smooth)
IK_SPEED      = 0.40   # m/s
FK_SPEED      = 0.80   # rad/s
WHEEL_RADIUS  = 0.090  # m
K_ACCEL       = 3.0    # m/s² 가속
K_BRAKE       = 6.0    # m/s² 제동
K_YAW_ACCEL   = 4.0
K_YAW_BRAKE   = 8.0
IK_WORKSPACE  = 0.78   # m (어깨 기준 최대 도달거리)
EMA_ALPHA     = 0.05

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


def _accel(cur: float, tgt: float, ac: float, br: float, dt: float) -> float:
    if abs(tgt) > 1e-4:
        diff = tgt - cur
        return cur + math.copysign(min(abs(diff), ac * dt), diff)
    step = br * dt
    return math.copysign(max(0.0, abs(cur) - step), cur) if cur != 0.0 else 0.0


class TeleopController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m = model
        self.d = data
        self.ks = KeyState()

        # ── 헬퍼 ────────────────────────────────────────────────────────
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

        # ── Actuator IDs ────────────────────────────────────────────────
        self._a_steer = {k: aid(f'{k}_wheel_steer') for k in WHEEL_XY}
        self._a_drive = {k: aid(f'{k}_wheel_drive') for k in WHEEL_XY}
        self._a_lift  = aid('lift_joint')
        self._a_arm_l = [aid(n) for n in ARM_L]
        self._a_arm_r = [aid(n) for n in ARM_R]
        self._a_fin_l = {n: try_aid(n) for n in FIN_L}
        self._a_fin_r = {n: try_aid(n) for n in FIN_R}

        # ── Joint 주소 ──────────────────────────────────────────────────
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

        # ── EE / shoulder body IDs ──────────────────────────────────────
        self._ee_l = bid('hx5_l_base')
        self._ee_r = bid('hx5_r_base')

        jl1_bid = model.jnt_bodyid[jid('arm_l_joint1')]
        jr1_bid = model.jnt_bodyid[jid('arm_r_joint1')]
        self._shoulder_l_bid = model.body_parentid[jl1_bid]
        self._shoulder_r_bid = model.body_parentid[jr1_bid]

        # ── Can reset ───────────────────────────────────────────────────
        can_j = try_jid('can_free')
        if can_j is not None:
            self._can_qadr = model.jnt_qposadr[can_j]
            self._can_vadr = model.jnt_dofadr[can_j]
        else:
            self._can_qadr = self._can_vadr = None
        self._can_init_qpos: np.ndarray | None = None

        # ── 속도 상태 (물리 베이스용) ───────────────────────────────────
        self._vx         = 0.0   # 원하는 base 속도 (world-frame)
        self._vy         = 0.0
        self._yaw_rate   = 0.0

        # ── IK 상태 ─────────────────────────────────────────────────────
        self._ik_tgt_l_base = np.zeros(3)
        self._ik_tgt_r_base = np.zeros(3)
        self._ik_err_l = 0.0
        self._ik_err_r = 0.0

        # ── FK 상태 ─────────────────────────────────────────────────────
        self._mode       = 'ik'   # 'ik' | 'fk'
        self._fk_joint   = 0      # 선택된 joint 인덱스 (0-6)
        self._fk_arm     = 'l'    # 'l' | 'r'

        # ── 그립 ────────────────────────────────────────────────────────
        self._grip_l  = False
        self._grip_r  = False
        self._tgl_l_t = -99.0
        self._tgl_r_t = -99.0

        # ── 기타 상태 ───────────────────────────────────────────────────
        self.show_gizmo  = True
        self._cam_follow = False
        self._fullscreen = False
        self._wall_t0    = time.perf_counter()
        self._freq_ema   = 60.0

        # ── Public attributes (markers.py 등에서 읽음) ─────────────────
        self.ik_world_tgt_l = np.zeros(3)
        self.ik_world_tgt_r = np.zeros(3)
        self.ee_pos_l       = np.zeros(3)
        self.ee_pos_r       = np.zeros(3)
        self.base_world_pos = np.zeros(3)

    # ── base_yaw (물리 quaternion에서 읽음) ─────────────────────────────

    @property
    def base_yaw(self) -> float:
        qa = self._fj_qpos
        qw = float(self.d.qpos[qa + 3])
        qx = float(self.d.qpos[qa + 4])
        qy = float(self.d.qpos[qa + 5])
        qz = float(self.d.qpos[qa + 6])
        return math.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        qa = self._fj_qpos
        self.d.qpos[qa + 2] = 0.1465   # 바퀴가 바닥에 닿는 높이
        self.d.qpos[qa + 3] = 1.0      # quaternion w=1 (yaw=0)
        mujoco.mj_forward(self.m, self.d)

        self._vx = self._vy = self._yaw_rate = 0.0

        self._ik_tgt_l_base = self._world_to_base(self.d.xpos[self._ee_l].copy())
        self._ik_tgt_r_base = self._world_to_base(self.d.xpos[self._ee_r].copy())
        self._grip_l = self._grip_r = False
        self._mode = 'ik'

        if self._can_qadr is not None:
            self._can_init_qpos = self.d.qpos[self._can_qadr: self._can_qadr + 7].copy()

        self._apply_grip('l', 0.0)
        self._apply_grip('r', 0.0)
        self._wall_t0 = time.perf_counter()

        # 바퀴 정지
        for k in WHEEL_XY:
            self.d.ctrl[self._a_steer[k]] = 0.0
            self.d.ctrl[self._a_drive[k]] = 0.0

    # ── 키 콜백 (GLFW 스레드) ────────────────────────────────────────────

    def on_key(self, key: int):
        self.ks.on_key(key)
        t = time.perf_counter()

        # 그립 토글
        if key == _K['Z'] and (t - self._tgl_l_t) > 0.3:
            self._grip_l = not self._grip_l; self._tgl_l_t = t
        if key == _K['X'] and (t - self._tgl_r_t) > 0.3:
            self._grip_r = not self._grip_r; self._tgl_r_t = t

        # 모드 전환
        if key == _K['TAB']:
            if self._mode == 'ik':
                self._mode = 'fk'
            else:
                # FK→IK: 현재 EE 위치로 IK 타겟 초기화
                self._ik_tgt_l_base = self._world_to_base(
                    self.d.xpos[self._ee_l].copy())
                self._ik_tgt_r_base = self._world_to_base(
                    self.d.xpos[self._ee_r].copy())
                self._mode = 'ik'

        # FK 조인트 선택
        if self._mode == 'fk':
            if key == _K['LBRACK']:
                self._fk_joint = (self._fk_joint - 1) % 7
            if key == _K['RBRACK']:
                self._fk_joint = (self._fk_joint + 1) % 7
            if key == _K['1']:
                self._fk_arm = 'l'
            if key == _K['2']:
                self._fk_arm = 'r'

        # 공통
        if key == _K['F']:
            self._cam_follow = not self._cam_follow
        if key == _K['G']:
            self.show_gizmo = not self.show_gizmo
        if key == _K['R']:
            self._reset_can()
        if key == _K['F11']:
            self._toggle_fullscreen()

    # ── 메인 업데이트 ────────────────────────────────────────────────────

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

    # ── 베이스 (물리 바퀴 actuator만 사용) ──────────────────────────────

    def _update_base(self, dt: float):
        ks = self.ks
        tvx  = (float(ks.is_down(_K['W'])) - float(ks.is_down(_K['S']))) * BASE_MAX_SPD
        tvy  = (float(ks.is_down(_K['A'])) - float(ks.is_down(_K['D']))) * BASE_MAX_SPD
        tyaw = (float(ks.is_down(_K['LEFT'])) - float(ks.is_down(_K['RIGHT']))) * YAW_MAX_SPD

        # 관성 모델 (속도 제한)
        self._vx       = _accel(self._vx,       tvx,  K_ACCEL,     K_BRAKE,     dt)
        self._vy       = _accel(self._vy,        tvy,  K_ACCEL,     K_BRAKE,     dt)
        self._yaw_rate = _accel(self._yaw_rate,  tyaw, K_YAW_ACCEL, K_YAW_BRAKE, dt)

        yaw = self.base_yaw
        c, s_ = math.cos(yaw), math.sin(yaw)

        # body-frame → world-frame 속도
        wx = c * self._vx - s_ * self._vy
        wy = s_ * self._vx + c  * self._vy

        # 각 바퀴의 원하는 속도 → steer각 + drive 각속도
        for name, wxy in WHEEL_XY.items():
            # 바퀴 중심에서의 속도 (world-frame)
            wvx = wx  - self._yaw_rate * wxy[1]
            wvy = wy  + self._yaw_rate * wxy[0]
            spd = math.sqrt(wvx ** 2 + wvy ** 2)

            ang  = math.atan2(wvy, wvx) if spd > 0.01 else 0.0
            sign = 1.0
            if ang > math.pi / 2:
                ang -= math.pi; sign = -1.0
            elif ang < -math.pi / 2:
                ang += math.pi; sign = -1.0

            # steer = 위치 서보 (라디안), drive = 각속도 서보 (rad/s)
            self.d.ctrl[self._a_steer[name]] = ang
            self.d.ctrl[self._a_drive[name]] = sign * spd / WHEEL_RADIUS

        # 공용 속성 업데이트
        qa = self._fj_qpos
        self.base_world_pos = np.array([
            float(self.d.qpos[qa]),
            float(self.d.qpos[qa + 1]),
            float(self.d.qpos[qa + 2]),
        ])

    # ── 리프트 (부드러운 연속 이동) ─────────────────────────────────────

    def _update_lift(self, dt: float):
        qa      = self._j_lift_qadr
        lo, hi  = self._j_lift_range
        ks = self.ks
        delta = (float(ks.is_down(_K['Q'])) - float(ks.is_down(_K['E']))) * LIFT_STEP
        if delta != 0.0:
            self.d.qpos[qa] = float(np.clip(self.d.qpos[qa] + delta, lo, hi))
        self.d.ctrl[self._a_lift] = self.d.qpos[qa]

    # ── IK 타겟 (base-frame, 연속 속도) ─────────────────────────────────

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

    # ── IK 솔버 ─────────────────────────────────────────────────────────

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

    # ── FK 직접 관절 제어 ────────────────────────────────────────────────

    def _update_fk(self, dt: float):
        # mj_step 이후 xpos가 갱신되므로 항상 EE 캐시 갱신
        self.ee_pos_l = self.d.xpos[self._ee_l].copy()
        self.ee_pos_r = self.d.xpos[self._ee_r].copy()
        self.ik_world_tgt_l = self.ee_pos_l
        self.ik_world_tgt_r = self.ee_pos_r

        ks = self.ks
        delta = (float(ks.is_down(_K['I'])) - float(ks.is_down(_K['K']))) * FK_SPEED * dt
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

    # ── 파지 ────────────────────────────────────────────────────────────

    def _update_grip(self):
        self._apply_grip('l', 1.0 if self._grip_l else 0.0)
        self._apply_grip('r', 1.0 if self._grip_r else 0.0)

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

    # ── 캔 리셋 ─────────────────────────────────────────────────────────

    def _reset_can(self):
        if self._can_qadr is None or self._can_init_qpos is None:
            return
        self.d.qpos[self._can_qadr: self._can_qadr + 7] = self._can_init_qpos
        self.d.qvel[self._can_vadr: self._can_vadr + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)

    # ── 전체화면 (GLFW 스레드에서 호출) ─────────────────────────────────

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
                    mode.size.width, mode.size.height, mode.refresh_rate)
                self._fullscreen = True
            else:
                _glfw.set_window_monitor(win, None, 80, 80, 1280, 720, 0)
                self._fullscreen = False
        except Exception:
            pass

    # ── 좌표 변환 ────────────────────────────────────────────────────────

    def _rot(self) -> np.ndarray:
        yaw = self.base_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _base_to_world(self, local: np.ndarray) -> np.ndarray:
        qa  = self._fj_qpos
        bp  = np.array([float(self.d.qpos[qa]),
                         float(self.d.qpos[qa + 1]),
                         float(self.d.qpos[qa + 2])])
        return self._rot() @ local + bp

    def _world_to_base(self, world: np.ndarray) -> np.ndarray:
        qa  = self._fj_qpos
        bp  = np.array([float(self.d.qpos[qa]),
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

    # ── 오버레이 ─────────────────────────────────────────────────────────

    def overlay(self, viewer):
        # 카메라 추적
        if self._cam_follow:
            try:
                qa = self._fj_qpos
                viewer.cam.lookat[0] = float(self.d.qpos[qa])
                viewer.cam.lookat[1] = float(self.d.qpos[qa + 1])
                viewer.cam.lookat[2] = float(self.d.qpos[qa + 2]) + 0.5
            except Exception:
                pass

        # 3D 마커
        try:
            from .markers import render as _rm
            _rm(viewer.user_scn, self)
        except Exception:
            pass

        self._draw_hud(viewer)

    def _draw_hud(self, viewer):
        el  = self._ik_err_l * 1000.0
        er  = self._ik_err_r * 1000.0

        def col(e):
            return 'G' if e < 5.0 else ('Y' if e < 20.0 else 'R')

        qa    = self._fj_qpos
        bx    = float(self.d.qpos[qa])
        by    = float(self.d.qpos[qa + 1])
        bz    = float(self.d.qpos[qa + 2])
        lz    = float(self.d.qpos[self._j_lift_qadr])
        yaw   = self.base_yaw
        deg   = math.degrees(yaw)
        vspd  = math.sqrt(self._vx ** 2 + self._vy ** 2)
        wall  = time.perf_counter() - self._wall_t0
        simtm = float(self.d.time)

        # ── 조인트 상태 패널 ─────────────────────────────────────────────
        def joint_bar(val, lo, hi, width=12):
            """범위 대비 현재값을 ASCII 바로 표시."""
            if hi <= lo:
                return '[' + '?' * width + ']'
            pct = (val - lo) / (hi - lo)
            pct = max(0.0, min(1.0, pct))
            filled = int(round(pct * width))
            bar = '=' * filled + '-' * (width - filled)
            return f'[{bar}]'

        def arm_row(qadrs, ranges, selected_idx, label):
            lines = [f'── {label} ──']
            for i in range(7):
                val    = math.degrees(float(self.d.qpos[qadrs[i]]))
                lo, hi = math.degrees(ranges[i][0]), math.degrees(ranges[i][1])
                bar    = joint_bar(float(self.d.qpos[qadrs[i]]),
                                   ranges[i][0], ranges[i][1])
                mark   = '►' if (self._mode == 'fk' and i == selected_idx) else ' '
                # 한계 경고
                pct = (float(self.d.qpos[qadrs[i]]) - ranges[i][0]) / (
                    ranges[i][1] - ranges[i][0] + 1e-10)
                warn = '!' if (pct < 0.05 or pct > 0.95) else ' '
                lines.append(
                    f'{mark}J{i+1} {bar} {val:+7.1f}° [{lo:+.0f}~{hi:+.0f}]{warn}')
            return '\n'.join(lines)

        fk_arm_idx = self._fk_joint if self._mode == 'fk' else -1
        fk_arm     = self._fk_arm

        topleft = (
            arm_row(self._jl_qadrs, self._jl_ranges,
                    fk_arm_idx if fk_arm == 'l' else -1, 'Left Arm') +
            '\n\n' +
            arm_row(self._jr_qadrs, self._jr_ranges,
                    fk_arm_idx if fk_arm == 'r' else -1, 'Right Arm') +
            f'\nLift {lz:+.3f} m'
        )

        # ── 텔레메트리 ───────────────────────────────────────────────────
        topright = (
            '─── Telemetry ───\n'
            f'Sim  {simtm:8.3f} s\n'
            f'Wall {wall:8.1f} s\n'
            f'Freq {self._freq_ema:6.1f} Hz\n'
            '─── IK Error ────\n'
            f'L {el:6.1f} mm [{col(el)}]\n'
            f'R {er:6.1f} mm [{col(er)}]'
        )

        # ── 상태 패널 ────────────────────────────────────────────────────
        mode_str = (
            f'[FK] Arm:{self._fk_arm.upper()} J{self._fk_joint+1}  '
            f'[/] joint select  I/K move'
            if self._mode == 'fk' else
            f'[IK] L({self._ik_tgt_l_base[0]:+.2f},{self._ik_tgt_l_base[1]:+.2f},'
            f'{self._ik_tgt_l_base[2]:+.2f})  '
            f'R({self._ik_tgt_r_base[0]:+.2f},{self._ik_tgt_r_base[1]:+.2f},'
            f'{self._ik_tgt_r_base[2]:+.2f})'
        )
        bottomleft = (
            f'Mode: {mode_str}\n'
            f'Base  pos=({bx:+.2f},{by:+.2f},{bz:+.2f})  '
            f'yaw={deg:+.1f}°  |v|={vspd:.2f} m/s\n'
            f'Grip  L={"GRIP" if self._grip_l else "open"}  '
            f'R={"GRIP" if self._grip_r else "open"}\n'
            f'Cam={"ON" if self._cam_follow else "off"}  '
            f'Gizmo={"ON" if self.show_gizmo else "off"}  '
            f'Fullscr={"ON" if self._fullscreen else "off"}'
        )

        bottomright = (
            '── Controls ──────────────\n'
            'WASD=이동  ←/→=yaw  Q/E=리프트\n'
            'Tab=FK↔IK 전환\n'
            'IK: IJKL UO=EE이동  1=좌팔  2=우팔\n'
            'FK: 1/2=팔선택  [/]=조인트  I/K=각도\n'
            'Z/X=그립  F=카메라  G=기즈모\n'
            'R=캔리셋  F11=전체화면'
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
                    bottomleft, bottomright,
                )])
            except Exception:
                pass
