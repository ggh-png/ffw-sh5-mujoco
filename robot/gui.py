"""
FFW-SH5 Dear PyGui 컨트롤 패널.

별도 데몬 스레드에서 실행.  controller 의 공개 속성에 직접 접근해
IK 타겟, FK 조인트 각도, 그립/엄지 값을 읽고 씁니다.

IMPORTANT: ControlPanel() 은 반드시 mujoco.viewer.launch_passive()
           컨텍스트 *안*에서 생성해야 합니다 (EGL 컨텍스트 충돌 방지).

DPG 콜백 규칙:
  DPG는 항상 (sender, app_data, user_data) 세 인수를 전달.
  lambda에서 sd=side 같은 기본값 패턴을 쓰면 DPG가 세 번째 인수를
  user_data로 전달해 기본값을 덮어쓴다 → 모든 캡처 변수가 None이 됨.
  올바른 패턴: 메서드의 세 번째 파라미터를 user_data로 받거나,
               widget 생성 시 user_data= 를 명시적으로 지정.
"""
import math
import threading
import time

import numpy as np

try:
    import dearpygui.dearpygui as dpg
    _HAS_DPG = True
except ImportError:
    _HAS_DPG = False

try:
    import mujoco as _mujoco
except ImportError:
    _mujoco = None


# ── Quaternion helpers ────────────────────────────────────────────────────

def _rpy_from_quat(q: np.ndarray):
    """[w,x,y,z] → (roll, pitch, yaw) radians."""
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll/Pitch/Yaw (rad, ZYX) → [w,x,y,z] quaternion."""
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


# ── Colour palette ────────────────────────────────────────────────────────
_C_BG     = (28,  28,  28)
_C_HEADER = (38,  79, 120)
_C_ACCENT = (86, 156, 214)
_C_OK     = (78, 201, 176)
_C_FRAME  = (55,  55,  55)
_C_L      = (100, 200, 255)
_C_R      = (100, 255, 200)
_C_WARN   = (255, 180,  80)


class ControlPanel:
    """GUI 컨트롤 패널 (daemon thread)."""

    _POLL_INTERVAL = 0.05   # display refresh period (20 Hz)
    _INIT_DELAY    = 1.5    # wait for MuJoCo GL context to settle

    def __init__(self, ctrl):
        self._ctrl = ctrl
        if not _HAS_DPG:
            print('[gui] dearpygui not available — panel disabled')
            return
        t = threading.Thread(target=self._mainloop, daemon=True, name='dpg-panel')
        t.start()

    # ── Dear PyGui main loop ──────────────────────────────────────────────

    def _mainloop(self):
        time.sleep(self._INIT_DELAY)

        dpg.create_context()
        dpg.create_viewport(
            title='FFW-SH5 Control Panel',
            width=520, height=1020,
            x_pos=12, y_pos=12,
            resizable=True,
            decorated=True,
        )
        dpg.setup_dearpygui()
        self._apply_theme()
        self._build_ui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            try:
                self._poll()
            except Exception:
                pass
            dpg.render_dearpygui_frame()
            time.sleep(self._POLL_INTERVAL)

        dpg.destroy_context()

    # ── Theme ─────────────────────────────────────────────────────────────

    def _apply_theme(self):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,        _C_BG)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,    _C_HEADER)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,       _C_ACCENT)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (110, 180, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,          _C_FRAME)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,   (72, 72, 72))
                dpg.add_theme_color(dpg.mvThemeCol_Button,           _C_FRAME)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,    _C_HEADER)
                dpg.add_theme_color(dpg.mvThemeCol_Header,           _C_HEADER)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,    (52, 100, 160))
                dpg.add_theme_color(dpg.mvThemeCol_Text,             (212, 212, 212))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,    4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,      6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,     6, 3)
        dpg.bind_theme(t)

    # ── Build all UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        with dpg.window(
            label='FFW-SH5 Control Panel',
            width=510, height=1010,
            pos=(0, 0),
            no_move=False,
            no_resize=False,
            no_close=True,
            no_scrollbar=False,
        ):
            self._build_ik_section()
            dpg.add_spacer(height=4)
            self._build_lift_section()
            dpg.add_spacer(height=4)
            self._build_fk_section()
            dpg.add_spacer(height=4)
            self._build_grip_section()
            dpg.add_spacer(height=4)
            self._build_joint_monitor()
            dpg.add_spacer(height=4)
            self._build_utils_section()
            dpg.add_separator()
            self._status = dpg.add_text('Initialising...', color=_C_OK)

    # ── IK Targets (Position + Orientation) ──────────────────────────────

    def _build_ik_section(self):
        with dpg.collapsing_header(label='IK Targets', default_open=True):
            dpg.add_text(
                '  I/K=fwd  J/L=lat  U/O=up  (hold 1=L only, 2=R only)',
                color=(160, 160, 160))
            dpg.add_text(
                '  9=Ori-IK  0=Palm-IK  3/4=Roll  5/6=Pitch  7/8=Yaw',
                color=(160, 160, 160))
            dpg.add_spacer(height=2)

            self._ik_sl  = {}
            self._ik_val = {}
            self._rpy_sl  = {}
            self._rpy_val = {}

            POS_AXES = [
                ('X fwd', 'x', -0.50, 1.50),
                ('Y lat', 'y', -1.00, 1.00),
                ('Z  up', 'z',  0.00, 2.00),
            ]
            ORI_AXES = [
                ('Roll ', 'r', -180.0, 180.0),
                ('Pitch', 'p',  -90.0,  90.0),
                ('Yaw  ', 'y', -180.0, 180.0),
            ]

            for side, side_label, color in [
                    ('l', '[L] Arm', _C_L),
                    ('r', '[R] Arm', _C_R)]:
                dpg.add_text(f'  {side_label}  — Position', color=color)
                for axis_label, axis, lo, hi in POS_AXES:
                    key = f'{side}_{axis}'
                    try:
                        tgt  = (self._ctrl._ik_tgt_l_base if side == 'l'
                                else self._ctrl._ik_tgt_r_base)
                        init = float(tgt[('x', 'y', 'z').index(axis)])
                    except Exception:
                        init = 0.0

                    with dpg.table(header_row=False, borders_innerV=False,
                                   borders_outerH=False, borders_outerV=False):
                        dpg.add_table_column(width_fixed=True,   init_width_or_weight=60)
                        dpg.add_table_column(width_stretch=True, init_width_or_weight=0.8)
                        dpg.add_table_column(width_fixed=True,   init_width_or_weight=82)
                        with dpg.table_row():
                            dpg.add_text(f'   {axis_label}')
                            sl = dpg.add_slider_float(
                                default_value=init,
                                min_value=lo, max_value=hi,
                                width=-1,
                                callback=self._on_ik,
                                user_data=(side, axis),
                                format='',
                                no_input=False,
                            )
                            self._ik_sl[key] = sl
                            vl = dpg.add_text(f'{init:+.3f} m')
                            self._ik_val[key] = vl

                # Orientation sliders (deg)
                dpg.add_text(f'  {side_label}  — Orientation  (키 9로 활성화)', color=color)
                try:
                    q_init = (self._ctrl._ik_tgt_l_quat if side == 'l'
                              else self._ctrl._ik_tgt_r_quat)
                    r0, p0, y0 = _rpy_from_quat(q_init)
                    rpy_init = [math.degrees(r0), math.degrees(p0), math.degrees(y0)]
                except Exception:
                    rpy_init = [0.0, 0.0, 0.0]

                for i, (axis_label, axis, lo, hi) in enumerate(ORI_AXES):
                    key = f'{side}_{axis}_ori'
                    with dpg.table(header_row=False, borders_innerV=False,
                                   borders_outerH=False, borders_outerV=False):
                        dpg.add_table_column(width_fixed=True,   init_width_or_weight=60)
                        dpg.add_table_column(width_stretch=True, init_width_or_weight=0.8)
                        dpg.add_table_column(width_fixed=True,   init_width_or_weight=82)
                        with dpg.table_row():
                            dpg.add_text(f'   {axis_label}')
                            sl = dpg.add_slider_float(
                                default_value=rpy_init[i],
                                min_value=lo, max_value=hi,
                                width=-1,
                                callback=self._on_rpy,
                                user_data=(side, axis),
                                format='',
                                no_input=False,
                            )
                            self._rpy_sl[key] = sl
                            vl = dpg.add_text(f'{rpy_init[i]:+.1f} °')
                            self._rpy_val[key] = vl

                dpg.add_spacer(height=2)

            # Palm IK 토글 버튼 (오른팔 기준)
            dpg.add_spacer(height=2)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label='  Ori-IK ON/OFF (9)  ',
                    callback=self._toggle_ori_ik,
                )
                dpg.add_button(
                    label='  Palm-IK ON/OFF (0)  ',
                    callback=self._toggle_palm_ik,
                )
            self._ori_status  = dpg.add_text('Ori-IK: off', color=_C_WARN)
            self._palm_status = dpg.add_text('Palm-IK: off', color=_C_WARN)

    def _on_ik(self, _sender, value, user_data):
        side, axis = user_data
        idx = ('x', 'y', 'z').index(axis)
        tgt = (self._ctrl._ik_tgt_l_base if side == 'l'
               else self._ctrl._ik_tgt_r_base)
        tgt[idx] = float(value)

    def _on_rpy(self, _sender, value_deg, user_data):
        """Orientation slider 이동 → 해당 축 RPY 변경 후 쿼터니언 업데이트."""
        side, axis = user_data
        value_rad = math.radians(float(value_deg))

        if side == 'l':
            q = self._ctrl._ik_tgt_l_quat.copy()
        else:
            q = self._ctrl._ik_tgt_r_quat.copy()

        r, p, y = _rpy_from_quat(q)
        rpy = [r, p, y]
        rpy[('r', 'p', 'y').index(axis)] = value_rad
        new_q = _rpy_to_quat(rpy[0], rpy[1], rpy[2])

        if side == 'l':
            self._ctrl._ik_tgt_l_quat = new_q
        else:
            self._ctrl._ik_tgt_r_quat = new_q
        # 슬라이더를 움직이면 자동으로 Orientation IK 활성화
        self._ctrl._use_ori_ik = True

    def _toggle_ori_ik(self, _s=None, _a=None, _u=None):
        self._ctrl._use_ori_ik = not self._ctrl._use_ori_ik
        if self._ctrl._use_ori_ik:
            self._ctrl._ik_tgt_l_quat = self._ctrl.d.xquat[self._ctrl._ee_l].copy()
            self._ctrl._ik_tgt_r_quat = self._ctrl.d.xquat[self._ctrl._ee_r].copy()

    def _toggle_palm_ik(self, _s=None, _a=None, _u=None):
        ctrl = self._ctrl
        ctrl._use_palm_ik = not ctrl._use_palm_ik
        if ctrl._mode == 'ik':
            if ctrl._use_palm_ik:
                ctrl._ik_tgt_l_base = ctrl._world_to_base(ctrl._palm_center('l'))
                ctrl._ik_tgt_r_base = ctrl._world_to_base(ctrl._palm_center('r'))
            else:
                ctrl._ik_tgt_l_base = ctrl._world_to_base(ctrl.d.xpos[ctrl._ee_l].copy())
                ctrl._ik_tgt_r_base = ctrl._world_to_base(ctrl.d.xpos[ctrl._ee_r].copy())

    # ── Lift (로봇 높낮이) ────────────────────────────────────────────────

    def _build_lift_section(self):
        with dpg.collapsing_header(label='Lift — 로봇 높낮이  (Q=up  E=down)', default_open=True):
            lo, hi = self._ctrl._j_lift_range
            lo_mm  = lo * 1000.0
            hi_mm  = hi * 1000.0
            cur_mm = self._ctrl._lift_des * 1000.0

            with dpg.table(header_row=False, borders_innerV=False,
                           borders_outerH=False, borders_outerV=False):
                dpg.add_table_column(width_fixed=True,   init_width_or_weight=60)
                dpg.add_table_column(width_stretch=True, init_width_or_weight=0.8)
                dpg.add_table_column(width_fixed=True,   init_width_or_weight=82)
                with dpg.table_row():
                    dpg.add_text('  Lift')
                    self._lift_sl = dpg.add_slider_float(
                        default_value=cur_mm,
                        min_value=lo_mm, max_value=hi_mm,
                        width=-1,
                        callback=self._on_lift,
                        format='',
                        no_input=False,
                    )
                    self._lift_val = dpg.add_text(f'{cur_mm:+.0f} mm')

            dpg.add_spacer(height=2)
            with dpg.group(horizontal=True):
                dpg.add_button(label='  Top   ',
                               callback=self._set_lift, user_data=hi_mm)
                dpg.add_button(label='  Mid   ',
                               callback=self._set_lift, user_data=(lo_mm + hi_mm) / 2)
                dpg.add_button(label=' Bottom ',
                               callback=self._set_lift, user_data=lo_mm)

    def _on_lift(self, _s, value_mm, _u=None):
        self._ctrl._lift_des = float(value_mm) / 1000.0

    def _set_lift(self, _s, _a, user_data):
        val_mm = float(user_data)
        self._ctrl._lift_des = val_mm / 1000.0
        dpg.set_value(self._lift_sl, val_mm)

    # ── FK Joints ─────────────────────────────────────────────────────────

    def _build_fk_section(self):
        with dpg.collapsing_header(label='FK Joints', default_open=False):
            dpg.add_text(
                '  Tab→FK mode  1/2=arm  [/]=joint  I/K=±angle',
                color=(160, 160, 160))
            dpg.add_text(
                '  Sliders only take effect in FK mode (Tab to switch)',
                color=_C_OK)
            dpg.add_spacer(height=2)

            with dpg.group(horizontal=True):
                dpg.add_text('  Arm: ')
                self._arm_rb = dpg.add_radio_button(
                    ['Left', 'Right'],
                    default_value='Left',
                    callback=self._on_arm_change,
                    horizontal=True,
                )
            dpg.add_spacer(height=4)

            self._jnt_sl  = []
            self._jnt_val = []

            ranges = self._ctrl._jl_ranges
            for i in range(7):
                lo_d = math.degrees(ranges[i][0])
                hi_d = math.degrees(ranges[i][1])
                cur  = math.degrees(float(self._ctrl.d.qpos[self._ctrl._jl_qadrs[i]]))

                with dpg.table(header_row=False, borders_innerV=False,
                               borders_outerH=False, borders_outerV=False):
                    dpg.add_table_column(width_fixed=True,   init_width_or_weight=36)
                    dpg.add_table_column(width_stretch=True, init_width_or_weight=0.8)
                    dpg.add_table_column(width_fixed=True,   init_width_or_weight=66)
                    with dpg.table_row():
                        dpg.add_text(f' J{i+1}')
                        sl = dpg.add_slider_float(
                            default_value=cur,
                            min_value=lo_d, max_value=hi_d,
                            width=-1,
                            callback=self._on_joint,
                            user_data=i,
                            format='',
                        )
                        vl = dpg.add_text(f'{cur:+.1f}°')

                self._jnt_sl.append((sl, lo_d, hi_d))
                self._jnt_val.append(vl)

            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(label='  Zero All  ',
                               callback=self._zero_all_joints)
                dpg.add_button(label='  Reset Arm  ',
                               callback=self._reset_arm)

    def _current_arm(self) -> str:
        return 'l' if dpg.get_value(self._arm_rb) == 'Left' else 'r'

    def _on_arm_change(self, _sender, app_data, _user_data=None):
        arm    = 'l' if app_data == 'Left' else 'r'
        qadrs  = self._ctrl._jl_qadrs if arm == 'l' else self._ctrl._jr_qadrs
        ranges = self._ctrl._jl_ranges if arm == 'l' else self._ctrl._jr_ranges
        for i, (sl, _, _) in enumerate(self._jnt_sl):
            lo_d = math.degrees(ranges[i][0])
            hi_d = math.degrees(ranges[i][1])
            cur  = math.degrees(float(self._ctrl.d.qpos[qadrs[i]]))
            dpg.configure_item(sl, min_value=lo_d, max_value=hi_d)
            dpg.set_value(sl, max(lo_d, min(hi_d, cur)))
            self._jnt_sl[i] = (sl, lo_d, hi_d)

    def _on_joint(self, _sender, value_deg, joint_idx):
        angle  = math.radians(float(value_deg))
        arm    = self._current_arm()
        qadrs  = self._ctrl._jl_qadrs if arm == 'l' else self._ctrl._jr_qadrs
        aids   = self._ctrl._a_arm_l  if arm == 'l' else self._ctrl._a_arm_r
        ranges = self._ctrl._jl_ranges if arm == 'l' else self._ctrl._jr_ranges
        lo, hi = ranges[joint_idx]
        val = float(np.clip(angle, lo, hi))
        self._ctrl.d.qpos[qadrs[joint_idx]] = val
        self._ctrl.d.ctrl[aids[joint_idx]]  = val

    def _zero_all_joints(self, _s=None, _a=None, _u=None):
        arm    = self._current_arm()
        qadrs  = self._ctrl._jl_qadrs if arm == 'l' else self._ctrl._jr_qadrs
        aids   = self._ctrl._a_arm_l  if arm == 'l' else self._ctrl._a_arm_r
        ranges = self._ctrl._jl_ranges if arm == 'l' else self._ctrl._jr_ranges
        for i in range(7):
            lo, hi = ranges[i]
            v = float(np.clip(0.0, lo, hi))
            self._ctrl.d.qpos[qadrs[i]] = v
            self._ctrl.d.ctrl[aids[i]]  = v
            sl = self._jnt_sl[i][0]
            dpg.set_value(sl, math.degrees(v))

    def _reset_arm(self, _s=None, _a=None, _u=None):
        self._zero_all_joints()

    # ── Hand / Grip ───────────────────────────────────────────────────────

    def _build_grip_section(self):
        with dpg.collapsing_header(label='Hand / Grip', default_open=True):
            dpg.add_text(
                '  Z/C = L 3-finger   X/V = R 3-finger',
                color=(160, 160, 160))
            dpg.add_text(
                '  A/S = L thumb      H/N = R thumb',
                color=(160, 160, 160))
            dpg.add_spacer(height=4)

            self._grip_sl   = {}
            self._grip_val  = {}
            self._thumb_sl  = {}
            self._thumb_val = {}

            for side, label, color in [
                    ('l', 'Left  hand', _C_L),
                    ('r', 'Right hand', _C_R)]:

                dpg.add_text(f'  {label}', color=color)

                close_k = 'Z' if side == 'l' else 'X'
                open_k  = 'C' if side == 'l' else 'V'
                dpg.add_text(f'    3-finger  ({close_k}=close  {open_k}=open)',
                             color=(160, 160, 160))

                with dpg.table(header_row=False, borders_innerV=False,
                               borders_outerH=False, borders_outerV=False):
                    dpg.add_table_column(width_fixed=True,   init_width_or_weight=52)
                    dpg.add_table_column(width_stretch=True, init_width_or_weight=0.8)
                    dpg.add_table_column(width_fixed=True,   init_width_or_weight=46)
                    with dpg.table_row():
                        dpg.add_text('  grasp')
                        sl = dpg.add_slider_float(
                            default_value=0.0, min_value=0.0, max_value=1.0,
                            width=-1, format='',
                            callback=self._on_grip,
                            user_data=side,
                        )
                        vl = dpg.add_text('  0%')

                self._grip_sl[side]  = sl
                self._grip_val[side] = vl

                with dpg.group(horizontal=True):
                    dpg.add_button(label='  GRIP  ',
                                   callback=self._set_grip,
                                   user_data=(side, 1.0))
                    dpg.add_button(label='  OPEN  ',
                                   callback=self._set_grip,
                                   user_data=(side, 0.0))
                dpg.add_spacer(height=2)

                t_close = 'A' if side == 'l' else 'H'
                t_open  = 'S' if side == 'l' else 'N'
                dpg.add_text(f'    thumb  ({t_close}=close  {t_open}=open)',
                             color=(160, 160, 160))

                with dpg.table(header_row=False, borders_innerV=False,
                               borders_outerH=False, borders_outerV=False):
                    dpg.add_table_column(width_fixed=True,   init_width_or_weight=52)
                    dpg.add_table_column(width_stretch=True, init_width_or_weight=0.8)
                    dpg.add_table_column(width_fixed=True,   init_width_or_weight=46)
                    with dpg.table_row():
                        dpg.add_text('  thumb')
                        tsl = dpg.add_slider_float(
                            default_value=0.0, min_value=0.0, max_value=1.0,
                            width=-1, format='',
                            callback=self._on_thumb,
                            user_data=side,
                        )
                        tvl = dpg.add_text('  0%')

                self._thumb_sl[side]  = tsl
                self._thumb_val[side] = tvl

                with dpg.group(horizontal=True):
                    dpg.add_button(label='  CURL  ',
                                   callback=self._set_thumb,
                                   user_data=(side, 1.0))
                    dpg.add_button(label='  FLAT  ',
                                   callback=self._set_thumb,
                                   user_data=(side, 0.0))
                dpg.add_spacer(height=6)

    def _on_grip(self, _s, value, side):
        if side == 'l':
            self._ctrl._grip_l = float(value)
        else:
            self._ctrl._grip_r = float(value)

    def _on_thumb(self, _s, value, side):
        if side == 'l':
            self._ctrl._thumb_l = float(value)
        else:
            self._ctrl._thumb_r = float(value)

    def _set_grip(self, _s, _a, user_data):
        side, value = user_data
        dpg.set_value(self._grip_sl[side], value)
        self._on_grip(None, value, side)

    def _set_thumb(self, _s, _a, user_data):
        side, value = user_data
        dpg.set_value(self._thumb_sl[side], value)
        self._on_thumb(None, value, side)

    # ── Joint Position Monitor ────────────────────────────────────────────

    def _build_joint_monitor(self):
        with dpg.collapsing_header(label='Joint Position Monitor', default_open=True):
            self._jmon_items = []   # (text_tag, qadr, lo, hi, name)

            if _mujoco is None:
                dpg.add_text('mujoco not available', color=_C_WARN)
                return

            ctrl = self._ctrl
            m    = ctrl.m

            with dpg.child_window(height=420, border=False, horizontal_scrollbar=False):
                # 관심 있는 조인트 그룹별로 표시
                groups = [
                    ('Wheels', [
                        'left_wheel_steer',  'left_wheel_drive',
                        'right_wheel_steer', 'right_wheel_drive',
                        'rear_wheel_steer',  'rear_wheel_drive',
                    ]),
                    ('Head', ['head_joint1', 'head_joint2']),
                    ('Left Arm', [f'arm_l_joint{i}' for i in range(1, 8)]),
                    ('Right Arm', [f'arm_r_joint{i}' for i in range(1, 8)]),
                    ('Left Fingers', [f'finger_l_joint{i}' for i in range(1, 21)]),
                    ('Right Fingers', [f'finger_r_joint{i}' for i in range(1, 21)]),
                ]

                for group_name, names in groups:
                    dpg.add_text(f'  — {group_name} —', color=_C_ACCENT)
                    for jname in names:
                        jid = _mujoco.mj_name2id(m, _mujoco.mjtObj.mjOBJ_JOINT, jname)
                        if jid < 0:
                            continue
                        jtype = m.jnt_type[jid]
                        if jtype == _mujoco.mjtJoint.mjJNT_FREE:
                            continue
                        qadr    = int(m.jnt_qposadr[jid])
                        lo, hi  = float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])
                        cur_deg = math.degrees(float(ctrl.d.qpos[qadr]))
                        # 이름을 24자로 잘라서 고정폭 표시
                        display_name = f'{jname:<26}'
                        t = dpg.add_text(
                            f'{display_name} {cur_deg:+7.2f}°',
                            color=(190, 190, 190))
                        self._jmon_items.append((t, qadr, lo, hi, jname))
                    dpg.add_spacer(height=2)

    # ── Utils (can respawn, reset) ────────────────────────────────────────

    def _build_utils_section(self):
        with dpg.collapsing_header(label='Utils', default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label='  Respawn Can  (R)  ',
                    callback=self._respawn_can,
                )
                dpg.add_button(
                    label='  Reset Robot  ',
                    callback=self._reset_robot,
                )

    def _respawn_can(self, _s=None, _a=None, _u=None):
        self._ctrl._reset_can()

    def _reset_robot(self, _s=None, _a=None, _u=None):
        self._ctrl.reset()

    # ── Poll: refresh display values from simulation ──────────────────────

    def _poll(self):
        ctrl = self._ctrl

        # IK position sliders + value labels
        for key, sl in self._ik_sl.items():
            side = key[0]
            axis = key[2]
            idx  = ('x', 'y', 'z').index(axis)
            tgt  = ctrl._ik_tgt_l_base if side == 'l' else ctrl._ik_tgt_r_base
            val  = float(tgt[idx])
            if not dpg.is_item_active(sl):
                dpg.set_value(sl, val)
            dpg.set_value(self._ik_val[key], f'{val:+.3f} m')

        # IK orientation (RPY) sliders + value labels
        for key, sl in self._rpy_sl.items():
            side = key[0]
            axis = key[2]   # 'r' / 'p' / 'y' (from '{side}_{axis}_ori')
            q = ctrl._ik_tgt_l_quat if side == 'l' else ctrl._ik_tgt_r_quat
            try:
                r, p, y = _rpy_from_quat(q)
                rpy_deg = [math.degrees(r), math.degrees(p), math.degrees(y)]
                val_deg = rpy_deg[('r', 'p', 'y').index(axis)]
            except Exception:
                val_deg = 0.0
            if not dpg.is_item_active(sl):
                dpg.set_value(sl, val_deg)
            dpg.set_value(self._rpy_val[key], f'{val_deg:+.1f} °')

        # Ori/Palm IK status
        dpg.set_value(
            self._ori_status,
            f'Ori-IK: {"ON" if ctrl._use_ori_ik else "off"}',
        )
        dpg.set_value(
            self._palm_status,
            f'Palm-IK: {"ON" if ctrl._use_palm_ik else "off"}',
        )

        # Lift slider
        cur_mm = ctrl._lift_des * 1000.0
        if not dpg.is_item_active(self._lift_sl):
            dpg.set_value(self._lift_sl, cur_mm)
        dpg.set_value(self._lift_val, f'{cur_mm:+.0f} mm')

        # FK joint sliders + value labels
        arm   = self._current_arm()
        qadrs = ctrl._jl_qadrs if arm == 'l' else ctrl._jr_qadrs
        for i, (sl, lo_d, hi_d) in enumerate(self._jnt_sl):
            deg = math.degrees(float(ctrl.d.qpos[qadrs[i]]))
            if not dpg.is_item_active(sl):
                dpg.set_value(sl, max(lo_d, min(hi_d, deg)))
            dpg.set_value(self._jnt_val[i], f'{deg:+.1f}°')

        # Grip + thumb
        for side in ('l', 'r'):
            grip  = ctrl._grip_l  if side == 'l' else ctrl._grip_r
            thumb = ctrl._thumb_l if side == 'l' else ctrl._thumb_r
            sl  = self._grip_sl[side]
            tsl = self._thumb_sl[side]
            if not dpg.is_item_active(sl):
                dpg.set_value(sl, grip)
            if not dpg.is_item_active(tsl):
                dpg.set_value(tsl, thumb)
            dpg.set_value(self._grip_val[side],  f'{int(grip  * 100):3d}%')
            dpg.set_value(self._thumb_val[side], f'{int(thumb * 100):3d}%')

        # Joint position monitor (20 Hz → 각 poll마다 전체 갱신)
        for t, qadr, lo, hi, jname in self._jmon_items:
            val     = float(ctrl.d.qpos[qadr])
            deg     = math.degrees(val)
            pct     = (val - lo) / (hi - lo) * 100.0 if hi > lo else 0.0
            pct     = max(0.0, min(100.0, pct))
            name26  = f'{jname:<26}'
            dpg.set_value(t, f'{name26} {deg:+7.2f}° ({pct:3.0f}%)')

        # Status bar
        el   = ctrl._ik_err_l * 1000.0
        er   = ctrl._ik_err_r * 1000.0
        freq = ctrl._freq_ema
        mode = ctrl._mode.upper()
        bx   = float(ctrl.d.qpos[ctrl._fj_qpos])
        by   = float(ctrl.d.qpos[ctrl._fj_qpos + 1])
        dpg.set_value(
            self._status,
            f'Mode:{mode}  IK-L:{el:.1f}mm  IK-R:{er:.1f}mm'
            f'  {freq:.0f}Hz  Base:({bx:.2f},{by:.2f})',
        )
