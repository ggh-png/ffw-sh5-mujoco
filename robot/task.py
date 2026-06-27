"""
FFW-SH5 자율 캔 집기 & 따르기 State Machine.

States
------
  IDLE → DETECT → PLAN_GRASP → APPROACH
       → PRE_REACH → REACH → GRASP → LIFT → HOME → POUR → DONE

핵심 설계
---------
  • IK 목표를 매 프레임 _IK_STEP씩 점진적으로 전진 (local minimum 방지)
  • 접근 경로는 캔 상단(z≈0.55) 위를 유지 → EE가 캔과 충돌하지 않음
  • GRASP 진입 시 캔을 EE에 키네마틱으로 고정(weld offset) → 물리 파지 불확실성 제거

기하 참고
---------
  로봇 원점 x=0, 테이블 x=0.80, 캔 (0.80, 0, ~0.495), 캔 top z≈0.550
  리프트=0    → 어깨 z≈1.44  (팔 닿지 않음)
  리프트=-0.5 → 어깨 z≈0.93  + base_x≥0.20 이면 도달 가능
"""
import numpy as np
import mujoco

# ── 튜닝 파라미터 ──────────────────────────────────────────────────────────
_CAN_BODY         = 'can'

# --- 파지 기하 ---
# EE 기준점(hx5_r_base) 위치. 손바닥이 캔에서 8cm 거리, 7cm 높이에 위치.
# 손가락이 앞으로 10~15cm 뻗어 캔을 감쌀 수 있음.
_GRASP_PALM_DIST  = 0.08             # 손바닥↔캔 중심 수평 거리 (m)
_GRASP_Z_OFF      = 0.065            # 캔 중심 위 높이 오프셋 (m, 캔 top=0.055 이상)
_PREGRASP_RETREAT = 0.13             # 파지점 뒤 수평 후퇴 거리 (m)
_PREGRASP_Z_EXTRA = 0.025            # 접근 준비점 추가 Z 오프셋 (m)
_LIFT_DELTA       = 0.22             # 파지 후 들어올리기 (m)

# --- 접근 이동 ---
_APPROACH_X       = 0.22             # 베이스 목표 X 위치 (m)
_APPROACH_LIFT    = -0.46            # 리프트 목표 위치 (m, 범위 [-0.5, 0])
_APPROACH_SPD     = 0.020            # 베이스 이동 속도 (m/frame)
_LIFT_SPD         = 0.004            # 리프트 하강 속도 (m/frame)

# --- 홈 위치 (로봇 베이스 기준, m) ---
_HOME_BASE        = np.array([0.40, -0.20, 0.62])

# --- IK 추종 ---
_IK_STEP          = 0.008            # IK 목표 전진 속도 (m/frame @ 60Hz)
_ARRIVE_TOL       = 0.040            # 도달 판정 거리 (m) — 넓게 설정해 안정성 확보
_TGT_TOL          = 0.003            # IK 목표 도달 판정 거리 (m)

# --- 파지 ---
_GRASP_TICKS      = 90               # 파지 대기 틱 (~1.5s @ 60Hz)

# --- 따르기 ---
# arm_r_joint6 (idx=5). 실제 관절 기능 확인 후 튜닝.
_POUR_JOINT       = 5                # 0-indexed
_POUR_DELTA       = np.radians(100)
_POUR_TICKS       = 180

# --- 타임아웃 ---
_MOVE_TIMEOUT     = 900              # ~15s @ 60Hz

# ── 상태 목록 ─────────────────────────────────────────────────────────────
STATES = [
    'IDLE', 'DETECT', 'PLAN_GRASP', 'APPROACH',
    'PRE_REACH', 'REACH', 'GRASP',
    'LIFT', 'HOME', 'POUR', 'DONE',
]

_MOVE_STATES = {'APPROACH', 'PRE_REACH', 'REACH', 'LIFT', 'HOME'}
# GRASP 이후 캔을 키네마틱으로 EE에 부착하는 상태들
_ATTACHED_STATES = {'GRASP', 'LIFT', 'HOME', 'POUR', 'DONE'}


class CanPourTask:
    """캔 집기 → 따르기 자율 State Machine."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m     = model
        self.d     = data
        self.state = 'IDLE'
        self._tick = 0

        self._can_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _CAN_BODY)
        if self._can_bid < 0:
            raise RuntimeError(f'body "{_CAN_BODY}" not found in model')

        self.can_pos       : np.ndarray | None = None
        self.grasp_pos     : np.ndarray | None = None
        self.pre_grasp_pos : np.ndarray | None = None
        self.lift_pos      : np.ndarray | None = None

        # 키네마틱 그립 부착 상태
        self._grasp_attached = False
        self._can_ee_offset  = np.zeros(3)   # can = EE + offset

    # ── 공개 API ──────────────────────────────────────────────────────────

    def trigger(self) -> None:
        if self.state in ('IDLE', 'DONE'):
            self._log('태스크 시작 → DETECT')
            self._grasp_attached = False
            self.state = 'DETECT'
            self._tick = 0
        else:
            self._log(f'태스크 중단 (이전: {self.state})')
            self._grasp_attached = False
            self.state = 'IDLE'

    def is_active(self) -> bool:
        return self.state not in ('IDLE', 'DONE')

    @property
    def suppress_ik_r(self) -> bool:
        return self.state == 'POUR'

    def step(self, ctrl) -> None:
        """렌더 프레임마다 한 번 호출."""
        d = ctrl.d

        if self.state == 'IDLE' or self.state == 'DONE':
            return

        # ── 키네마틱 캔 부착 유지 ────────────────────────────────────────
        if self._grasp_attached and self.state in _ATTACHED_STATES:
            self._apply_can_attachment(ctrl)

        if self.state == 'DETECT':
            self.can_pos = d.xpos[self._can_bid].copy()
            self._log(f'캔 감지: {np.round(self.can_pos, 3)}')
            self.state = 'PLAN_GRASP'

        elif self.state == 'PLAN_GRASP':
            self._compute_grasp(ctrl)
            ctrl._ik_tgt_r_base = ctrl._world_to_base(d.xpos[ctrl._ee_r].copy())
            ctrl._grip_r = 0.0
            self._tick   = 0
            self._log(
                f'파지점:  {np.round(self.grasp_pos, 3)}\n'
                f'       접근점: {np.round(self.pre_grasp_pos, 3)}\n'
                f'       → APPROACH'
            )
            self.state = 'APPROACH'

        elif self.state == 'APPROACH':
            self._step_approach(ctrl)

        elif self.state == 'PRE_REACH':
            self._advance_ik(ctrl, self.pre_grasp_pos)
            self._tick += 1
            if self._at_goal(ctrl, self.pre_grasp_pos):
                self._log('접근점 도달 → REACH')
                self._tick = 0
                self.state = 'REACH'
            elif self._tick > _MOVE_TIMEOUT:
                self._abort('PRE_REACH 타임아웃')

        elif self.state == 'REACH':
            self._advance_ik(ctrl, self.grasp_pos)
            self._tick += 1
            if self._at_goal(ctrl, self.grasp_pos):
                ctrl._grip_r = 1.0
                self._attach_can(ctrl)   # 캔을 EE에 키네마틱 부착
                self._log('파지점 도달 → GRASP')
                self._tick = 0
                self.state = 'GRASP'
            elif self._tick > _MOVE_TIMEOUT:
                self._abort('REACH 타임아웃')

        elif self.state == 'GRASP':
            ctrl._ik_tgt_r_base = ctrl._world_to_base(self.grasp_pos)
            ctrl._grip_r = 1.0
            self._tick  += 1
            if self._tick >= _GRASP_TICKS:
                self.lift_pos = self.grasp_pos + np.array([0.0, 0.0, _LIFT_DELTA])
                self._log(f'파지 완료 → LIFT  목표: {np.round(self.lift_pos, 3)}')
                self._tick = 0
                self.state = 'LIFT'

        elif self.state == 'LIFT':
            ctrl._grip_r = 1.0
            self._advance_ik(ctrl, self.lift_pos)
            self._tick += 1
            if self._at_goal(ctrl, self.lift_pos):
                self._log(f'리프트 완료 → HOME')
                self._tick = 0
                self.state = 'HOME'
            elif self._tick > _MOVE_TIMEOUT:
                self._abort('LIFT 타임아웃')

        elif self.state == 'HOME':
            ctrl._grip_r = 1.0
            home_world = ctrl._base_to_world(_HOME_BASE)
            self._advance_ik(ctrl, home_world)
            self._tick += 1
            if self._at_goal(ctrl, home_world):
                self._pour_start = float(d.qpos[ctrl._jr_qadrs[_POUR_JOINT]])
                self._log(f'홈 도달 → POUR  시작각: {np.degrees(self._pour_start):.1f}°')
                self._tick = 0
                self.state = 'POUR'
            elif self._tick > _MOVE_TIMEOUT:
                self._abort('HOME 타임아웃')

        elif self.state == 'POUR':
            ctrl._grip_r = 1.0
            t     = min(1.0, self._tick / _POUR_TICKS)
            angle = self._pour_start + _POUR_DELTA * t
            qadr  = ctrl._jr_qadrs[_POUR_JOINT]
            aid   = ctrl._a_arm_r[_POUR_JOINT]
            lo, hi = ctrl._jr_ranges[_POUR_JOINT]
            d.ctrl[aid] = float(np.clip(angle, lo, hi))
            self._tick  += 1
            if self._tick >= _POUR_TICKS:
                self._log('따르기 완료 → DONE')
                self.state = 'DONE'

    # ── IK 점진적 추종 ────────────────────────────────────────────────────

    def _advance_ik(self, ctrl, goal_world: np.ndarray) -> None:
        cur  = ctrl._base_to_world(ctrl._ik_tgt_r_base)
        to   = goal_world - cur
        dist = np.linalg.norm(to)
        if dist > _TGT_TOL:
            step = min(_IK_STEP, dist)
            ctrl._ik_tgt_r_base = ctrl._world_to_base(cur + to / dist * step)
        else:
            ctrl._ik_tgt_r_base = ctrl._world_to_base(goal_world)

    def _at_goal(self, ctrl, goal_world: np.ndarray) -> bool:
        tgt_world = ctrl._base_to_world(ctrl._ik_tgt_r_base)
        tgt_ok    = np.linalg.norm(tgt_world - goal_world) < _TGT_TOL
        ee_ok     = ctrl._ik_err_r < _ARRIVE_TOL
        return tgt_ok and ee_ok

    # ── 키네마틱 캔 부착 ─────────────────────────────────────────────────

    def _attach_can(self, ctrl) -> None:
        """GRASP 시점에 can-EE 오프셋을 고정."""
        if ctrl._can_qadr is None:
            return
        ee_pos  = ctrl.d.xpos[ctrl._ee_r].copy()
        can_pos = ctrl.d.xpos[self._can_bid].copy()
        self._can_ee_offset  = can_pos - ee_pos
        self._grasp_attached = True
        self._log(f'캔 부착  offset={np.round(self._can_ee_offset, 3)}')

    def _apply_can_attachment(self, ctrl) -> None:
        """매 프레임 can의 qpos를 EE+offset으로 강제 설정."""
        if ctrl._can_qadr is None:
            return
        ee_pos     = ctrl.d.xpos[ctrl._ee_r].copy()
        target_pos = ee_pos + self._can_ee_offset
        qa = ctrl._can_qadr
        ctrl.d.qpos[qa    : qa + 3] = target_pos   # xyz
        ctrl.d.qpos[qa + 3: qa + 7] = [1, 0, 0, 0] # upright
        ctrl.d.qvel[ctrl._can_vadr : ctrl._can_vadr + 6] = 0.0

    # ── APPROACH ──────────────────────────────────────────────────────────

    def _step_approach(self, ctrl) -> None:
        d  = ctrl.d
        qa = ctrl._fj_qpos
        da = ctrl._fj_dof

        ctrl._ik_tgt_r_base = ctrl._world_to_base(d.xpos[ctrl._ee_r].copy())

        for k in ctrl._a_drive:
            d.ctrl[ctrl._a_drive[k]] = 0.0
        ctrl._vx       = 0.0
        ctrl._yaw_rate = 0.0

        cur_x = float(d.qpos[qa])
        dx    = _APPROACH_X - cur_x
        if abs(dx) > 0.003:
            d.qpos[qa]     = cur_x + np.sign(dx) * min(_APPROACH_SPD, abs(dx))
            d.qvel[da]     = 0.0
            d.qvel[da + 1] = 0.0

        cur_lift = ctrl._lift_des
        dl       = _APPROACH_LIFT - cur_lift
        if abs(dl) > 0.001:
            new_lift = cur_lift + np.sign(dl) * min(_LIFT_SPD, abs(dl))
            ctrl._lift_des = new_lift   # _update_lift()가 qpos/ctrl에 반영

        base_ok = abs(float(d.qpos[qa]) - _APPROACH_X)             <= 0.005
        lift_ok = abs(float(d.qpos[ctrl._j_lift_qadr]) - _APPROACH_LIFT) <= 0.002

        self._tick += 1
        if base_ok and lift_ok:
            ctrl._ik_tgt_r_base = ctrl._world_to_base(self.pre_grasp_pos)
            self._log(
                f'접근 완료  base_x={d.qpos[qa]:.3f}  '
                f'lift={d.qpos[ctrl._j_lift_qadr]:.3f} → PRE_REACH'
            )
            self._tick = 0
            self.state = 'PRE_REACH'
        elif self._tick > _MOVE_TIMEOUT:
            self._abort('APPROACH 타임아웃')

    # ── 파지점 계산 ───────────────────────────────────────────────────────

    def _compute_grasp(self, ctrl) -> None:
        can  = self.can_pos
        base = ctrl.base_world_pos

        dir_xy = can[:2] - base[:2]
        dist   = np.linalg.norm(dir_xy)
        dir_xy = dir_xy / dist if dist > 1e-4 else np.array([1.0, 0.0])

        # 손바닥 기준점: 캔 중심에서 palm_dist만큼 떨어진 위치, 위로 Z오프셋
        # 손가락이 앞으로 뻗어 캔을 감싸는 구조 → EE가 캔에 충돌하지 않음
        grasp       = can.copy()
        grasp[:2]  -= dir_xy * _GRASP_PALM_DIST   # 손바닥↔캔 수평 거리
        grasp[2]   += _GRASP_Z_OFF                 # 캔 top 위 (충돌 회피)

        # 접근 준비점: 더 뒤에서 더 높은 위치에서 수평 진입
        pre_grasp      = grasp.copy()
        pre_grasp[:2] -= dir_xy * _PREGRASP_RETREAT
        pre_grasp[2]  += _PREGRASP_Z_EXTRA

        self.grasp_pos     = grasp
        self.pre_grasp_pos = pre_grasp

    # ── 기타 헬퍼 ────────────────────────────────────────────────────────

    def _abort(self, reason: str) -> None:
        self._log(f'중단: {reason} → IDLE')
        self._grasp_attached = False
        self.state = 'IDLE'

    @staticmethod
    def _log(msg: str) -> None:
        print(f'[task] {msg}')

    # ── HUD ──────────────────────────────────────────────────────────────

    def hud_text(self) -> str:
        try:
            cur_idx = STATES.index(self.state)
        except ValueError:
            cur_idx = 0

        lines = ['--- Can Pour Task  [P]=시작/중단 ---']
        for i, s in enumerate(STATES):
            mark = '>' if i == cur_idx else ('v' if i < cur_idx else ' ')
            lines.append(f' {mark} {s}')

        if self.can_pos is not None:
            c = self.can_pos
            lines.append(f' can ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})')

        if self.state in _MOVE_STATES:
            lines.append(f' tick {self._tick}')

        if self._grasp_attached:
            lines.append(' [CAN ATTACHED]')

        return '\n'.join(lines)

    # ── 시각화 ────────────────────────────────────────────────────────────

    @property
    def vis_target(self) -> 'np.ndarray | None':
        if self.state == 'PRE_REACH':
            return self.pre_grasp_pos
        if self.state in ('REACH', 'GRASP'):
            return self.grasp_pos
        if self.state == 'LIFT':
            return self.lift_pos
        return None
