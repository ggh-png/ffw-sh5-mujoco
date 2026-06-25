"""
FFW-SH5 Dear PyGui 컨트롤 패널.

별도 데몬 스레드에서 실행.  controller 의 공개 속성에 직접 접근해
IK 타겟, FK 조인트 각도, 그립 값을 읽고 씁니다.

IMPORTANT: ControlPanel() 은 반드시 mujoco.viewer.launch_passive()
           컨텍스트 *안*에서 생성해야 합니다 (EGL 컨텍스트 충돌 방지).
"""
import math
import threading
import time

try:
    import dearpygui.dearpygui as dpg
    _HAS_DPG = True
except ImportError:
    _HAS_DPG = False


# ── Colour palette ────────────────────────────────────────────────────────
_C_BG        = (28, 28, 28)
_C_HEADER    = (38, 79, 120)
_C_ACCENT    = (86, 156, 214)
_C_OK        = ( 78, 201, 176)
_C_WARN      = (220, 150,  50)
_C_FRAME     = (55,  55,  55)


class ControlPanel:
    """GUI 컨트롤 패널 (daemon thread).

    MuJoCo 뷰어 창이 생성된 뒤 (launch_passive with 블록 안)에서
    ControlPanel(ctrl) 을 호출해야 EGL 충돌이 없습니다.
    """

    _POLL_INTERVAL = 0.05   # seconds between display refreshes (20 Hz)
    _INIT_DELAY    = 1.5    # wait for MuJoCo GL context to settle

    def __init__(self, ctrl):
        self._ctrl  = ctrl
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
            width=500, height=700,
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
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,     _C_BG)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, _C_HEADER)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,    _C_ACCENT)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (110, 180, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,       _C_FRAME)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (72, 72, 72))
                dpg.add_theme_color(dpg.mvThemeCol_Button,        _C_FRAME)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  _C_HEADER)
                dpg.add_theme_color(dpg.mvThemeCol_Header,         _C_HEADER)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (52, 100, 160))
                dpg.add_theme_color(dpg.mvThemeCol_Text,          (212, 212, 212))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,    6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,   6, 3)
        dpg.bind_theme(t)

    # ── Build all UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        with dpg.window(
            label='FFW-SH5 Control Panel',
            width=490, height=690,
            pos=(0, 0),
            no_move=False,
            no_resize=False,
            no_close=True,
            no_scrollbar=False,
        ):
            self._build_ik_section()
            dpg.add_spacer(height=4)
            self._build_fk_section()
            dpg.add_spacer(height=4)
            self._build_grip_section()
            dpg.add_separator()
            self._status = dpg.add_text(
                'Initialising...', color=_C_OK)

    # ── IK Targets ────────────────────────────────────────────────────────

    def _build_ik_section(self):
        with dpg.collapsing_header(label='IK Targets', default_open=True):
            dpg.add_text(
                '  I/K=fwd  J/L=lat  U/O=up  (hold 1=L only, 2=R only)',
                color=(160, 160, 160))
            dpg.add_spacer(height=2)

            self._ik_sl  = {}   # key → slider tag
            self._ik_val = {}   # key → text tag

            AXES = [
                ('X fwd', 'x', -0.50, 1.50),
                ('Y lat', 'y', -1.00, 1.00),
                ('Z  up', 'z',  0.00, 2.00),
            ]

            for side, side_label, color in [
                    ('l', '[L] Arm', (100, 200, 255)),
                    ('r', '[R] Arm', (100, 255, 200))]:
                dpg.add_text(f'  {side_label}', color=color)
                for axis_label, axis, lo, hi in AXES:
                    key = f'{side}_{axis}'
                    try:
                        tgt = (self._ctrl._ik_tgt_l_base if side == 'l'
                               else self._ctrl._ik_tgt_r_base)
                        init = float(tgt[('x','y','z').index(axis)])
                    except Exception:
                        init = 0.0

                    with dpg.table(header_row=False, borders_innerV=False,
                                   borders_outerH=False, borders_outerV=False):
                        dpg.add_table_column(width_fixed=True,   init_width_or_weight=58)
                        dpg.add_table_column(width_stretch=True, init_width_or_weight=0.8)
                        dpg.add_table_column(width_fixed=True,   init_width_or_weight=82)

                        with dpg.table_row():
                            dpg.add_text(f'   {axis_label}')

                            sl = dpg.add_slider_float(
                                default_value=init,
                                min_value=lo, max_value=hi,
                                width=-1,
                                callback=lambda s, v, u=(side, axis): self._on_ik(u, v),
                                user_data=(side, axis),
                                format='',   # hide built-in label to avoid overlap
                                no_input=False,
                            )
                            self._ik_sl[key] = sl

                            vl = dpg.add_text(f'{init:+.3f} m')
                            self._ik_val[key] = vl

    def _on_ik(self, user_data, value):
        side, axis = user_data
        idx = ('x', 'y', 'z').index(axis)
        tgt = (self._ctrl._ik_tgt_l_base if side == 'l'
               else self._ctrl._ik_tgt_r_base)
        tgt[idx] = float(value)

    # ── FK Joints ─────────────────────────────────────────────────────────

    def _build_fk_section(self):
        with dpg.collapsing_header(label='FK Joints', default_open=True):
            dpg.add_text(
                '  Tab→FK  1/2=arm  [/]=joint  I/K=±angle',
                color=(160, 160, 160))
            dpg.add_text(
                '  Home=max  End=min  Del=zero',
                color=(180, 180, 100))
            dpg.add_spacer(height=2)

            with dpg.group(horizontal=True):
                dpg.add_text('  Arm: ')
                self._arm_rb = dpg.add_radio_button(
                    ['Left', 'Right'],
                    default_value=0,
                    callback=self._on_arm_change,
                    horizontal=True,
                )
            dpg.add_spacer(height=4)

            self._jnt_sl  = []   # list of (tag, lo_d, hi_d)
            self._jnt_val = []   # list of value text tags

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
                            callback=lambda s, v, idx=i: self._on_joint(idx, v),
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

    def _on_arm_change(self, sender, val):
        arm    = 'l' if val == 0 else 'r'
        qadrs  = self._ctrl._jl_qadrs if arm == 'l' else self._ctrl._jr_qadrs
        ranges = self._ctrl._jl_ranges if arm == 'l' else self._ctrl._jr_ranges
        for i, (sl, _, _) in enumerate(self._jnt_sl):
            lo_d = math.degrees(ranges[i][0])
            hi_d = math.degrees(ranges[i][1])
            cur  = math.degrees(float(self._ctrl.d.qpos[qadrs[i]]))
            dpg.configure_item(sl, min_value=lo_d, max_value=hi_d)
            dpg.set_value(sl, max(lo_d, min(hi_d, cur)))
            self._jnt_sl[i] = (sl, lo_d, hi_d)

    def _on_joint(self, joint_idx, value_deg):
        import numpy as np
        import mujoco
        angle  = math.radians(float(value_deg))
        arm    = 'l' if dpg.get_value(self._arm_rb) == 0 else 'r'
        qadrs  = self._ctrl._jl_qadrs if arm == 'l' else self._ctrl._jr_qadrs
        aids   = self._ctrl._a_arm_l  if arm == 'l' else self._ctrl._a_arm_r
        ranges = self._ctrl._jl_ranges if arm == 'l' else self._ctrl._jr_ranges
        lo, hi = ranges[joint_idx]
        clamped = float(np.clip(angle, lo, hi))
        self._ctrl.d.qpos[qadrs[joint_idx]] = clamped
        self._ctrl.d.ctrl[aids[joint_idx]]  = clamped
        mujoco.mj_forward(self._ctrl.m, self._ctrl.d)

    def _zero_all_joints(self):
        import numpy as np, mujoco
        arm    = 'l' if dpg.get_value(self._arm_rb) == 0 else 'r'
        qadrs  = self._ctrl._jl_qadrs if arm == 'l' else self._ctrl._jr_qadrs
        aids   = self._ctrl._a_arm_l  if arm == 'l' else self._ctrl._a_arm_r
        ranges = self._ctrl._jl_ranges if arm == 'l' else self._ctrl._jr_ranges
        for i in range(7):
            lo, hi = ranges[i]
            v = float(np.clip(0.0, lo, hi))
            self._ctrl.d.qpos[qadrs[i]] = v
            self._ctrl.d.ctrl[aids[i]]  = v
            sl, lo_d, hi_d = self._jnt_sl[i]
            dpg.set_value(sl, math.degrees(v))
        mujoco.mj_forward(self._ctrl.m, self._ctrl.d)

    def _reset_arm(self):
        self._zero_all_joints()

    # ── Hand / Grip ───────────────────────────────────────────────────────

    def _build_grip_section(self):
        with dpg.collapsing_header(label='Hand / Grip', default_open=True):
            dpg.add_text(
                '  Hold key: Z=L-close  C=L-open  X=R-close  V=R-open',
                color=(160, 160, 160))
            dpg.add_spacer(height=4)

            self._grip_sl  = {}
            self._grip_val = {}

            for side, label, close_key, open_key, color in [
                    ('l', 'Left  hand', 'Z', 'C', (100, 200, 255)),
                    ('r', 'Right hand', 'X', 'V', (100, 255, 200))]:
                dpg.add_text(f'  {label}  ({close_key}=close  {open_key}=open)',
                             color=color)

                with dpg.table(header_row=False, borders_innerV=False,
                               borders_outerH=False, borders_outerV=False):
                    dpg.add_table_column(width_fixed=True,   init_width_or_weight=52)
                    dpg.add_table_column(width_stretch=True, init_width_or_weight=0.8)
                    dpg.add_table_column(width_fixed=True,   init_width_or_weight=46)

                    with dpg.table_row():
                        dpg.add_text('  open')
                        sl = dpg.add_slider_float(
                            default_value=0.0,
                            min_value=0.0, max_value=1.0,
                            width=-1,
                            callback=lambda s, v, sd=side: self._on_grip(sd, v),
                            format='',
                        )
                        vl = dpg.add_text('  0%')

                self._grip_sl[side]  = sl
                self._grip_val[side] = vl

                with dpg.group(horizontal=True):
                    dpg.add_button(label='  GRIP  ',
                                   callback=lambda _, s=side: self._set_grip(s, 1.0))
                    dpg.add_button(label='  OPEN  ',
                                   callback=lambda _, s=side: self._set_grip(s, 0.0))
                dpg.add_spacer(height=6)

    def _on_grip(self, side, value):
        if side == 'l':
            self._ctrl._grip_l = float(value)
        else:
            self._ctrl._grip_r = float(value)

    def _set_grip(self, side, value):
        dpg.set_value(self._grip_sl[side], value)
        self._on_grip(side, value)

    # ── Poll: refresh display values from simulation ───────────────────────

    def _poll(self):
        ctrl = self._ctrl

        # IK target sliders + value labels
        for key, sl in self._ik_sl.items():
            side = key[0]
            axis = key[2]
            idx  = ('x', 'y', 'z').index(axis)
            tgt  = ctrl._ik_tgt_l_base if side == 'l' else ctrl._ik_tgt_r_base
            val  = float(tgt[idx])
            if not dpg.is_item_active(sl):
                dpg.set_value(sl, val)
            dpg.set_value(self._ik_val[key], f'{val:+.3f} m')

        # FK joint sliders + value labels
        arm    = 'l' if dpg.get_value(self._arm_rb) == 0 else 'r'
        qadrs  = ctrl._jl_qadrs if arm == 'l' else ctrl._jr_qadrs
        for i, (sl, lo_d, hi_d) in enumerate(self._jnt_sl):
            deg = math.degrees(float(ctrl.d.qpos[qadrs[i]]))
            if not dpg.is_item_active(sl):
                dpg.set_value(sl, max(lo_d, min(hi_d, deg)))
            dpg.set_value(self._jnt_val[i], f'{deg:+.1f}°')

        # Grip sliders + value labels
        for side in ('l', 'r'):
            grip = ctrl._grip_l if side == 'l' else ctrl._grip_r
            sl   = self._grip_sl[side]
            if not dpg.is_item_active(sl):
                dpg.set_value(sl, grip)
            dpg.set_value(self._grip_val[side], f'{int(grip*100):3d}%')

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
