"""
Dear PyGui 기반 컨트롤 패널.
별도 데몬 스레드에서 실행 — controller 의 공개 메서드로 상태를 읽고 씁니다.
"""
import math
import threading
import time

try:
    import dearpygui.dearpygui as dpg
    _HAS_DPG = True
except ImportError:
    _HAS_DPG = False


class ControlPanel:
    """FFW-SH5 GUI 컨트롤 패널 (daemon thread)."""

    POLL_HZ = 20   # display refresh rate

    def __init__(self, ctrl):
        self._ctrl = ctrl
        if not _HAS_DPG:
            print('[gui] dearpygui not available — skipping GUI panel')
            return
        t = threading.Thread(target=self._mainloop, daemon=True, name='gui-panel')
        t.start()

    # ── Dear PyGui main loop ──────────────────────────────────────────────

    def _mainloop(self):
        # Wait for MuJoCo viewer's GL context to be fully established before
        # creating Dear PyGui's own GLFW/GL context to avoid EGL conflicts.
        time.sleep(1.5)
        dpg.create_context()
        dpg.create_viewport(
            title='FFW-SH5 Control Panel',
            width=480, height=680,
            x_pos=10, y_pos=10,
            resizable=True,
        )
        dpg.setup_dearpygui()

        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,   (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (38, 79, 120))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,    (86, 156, 214))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (110, 180, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,    (60, 60, 60))
                dpg.add_theme_color(dpg.mvThemeCol_Button,     (60, 60, 60))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 100, 130))
                dpg.add_theme_color(dpg.mvThemeCol_Header,        (38, 79, 120))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (50, 100, 160))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 4)
        dpg.bind_theme(global_theme)

        self._build_ui()

        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            try:
                self._poll()
            except Exception:
                pass
            dpg.render_dearpygui_frame()
            time.sleep(1.0 / self.POLL_HZ)

        dpg.destroy_context()

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        ctrl = self._ctrl

        with dpg.window(label='FFW-SH5 Control Panel',
                        width=470, height=660,
                        pos=(0, 0),
                        no_move=False,
                        no_resize=False,
                        no_close=True):

            # ── IK Targets tab ──────────────────────────────────────────
            with dpg.collapsing_header(label='IK Targets', default_open=True):
                dpg.add_text('  Move EE: I/K=fwd  J/L=lat  U/O=up  (hold 1=L, 2=R)')
                dpg.add_separator()

                self._ik_sliders = {}
                self._ik_labels  = {}
                config = [
                    ('L', 'X fwd', 'l_x', -0.50, 1.50),
                    ('L', 'Y lat', 'l_y', -1.00, 1.00),
                    ('L', 'Z  up', 'l_z',  0.00, 2.00),
                    ('R', 'X fwd', 'r_x', -0.50, 1.50),
                    ('R', 'Y lat', 'r_y', -1.00, 1.00),
                    ('R', 'Z  up', 'r_z',  0.00, 2.00),
                ]
                cur_side = ''
                for side, axis_label, key, lo, hi in config:
                    if side != cur_side:
                        dpg.add_text(f'  [{side}] Arm', color=(100, 180, 255))
                        cur_side = side
                    with dpg.group(horizontal=True):
                        dpg.add_text(f'  {axis_label}', indent=10)
                        tag = f'ik_sl_{key}'
                        lbl = f'ik_lbl_{key}'
                        try:
                            init_val = float(ctrl._ik_tgt_l_base[['x', 'y', 'z'].index(key[2])]
                                             if key[0] == 'l'
                                             else ctrl._ik_tgt_r_base[['x', 'y', 'z'].index(key[2])])
                        except Exception:
                            init_val = 0.0
                        dpg.add_slider_float(
                            tag=tag,
                            default_value=init_val,
                            min_value=lo, max_value=hi,
                            width=220,
                            callback=lambda s, v, u=(side.lower(), key[2:], lo, hi): self._on_ik(u, v),
                            user_data=(side.lower(), key[2:], lo, hi),
                            format='%.3f m',
                        )
                        dpg.add_text('0.000 m', tag=lbl, indent=4)
                        self._ik_sliders[key] = tag
                        self._ik_labels[key]  = lbl

            dpg.add_spacer(height=4)

            # ── FK Joints tab ───────────────────────────────────────────
            with dpg.collapsing_header(label='FK Joints', default_open=True):
                dpg.add_text('  Adjust: Tab→FK  1/2=arm  [/]=joint  I/K=angle')
                dpg.add_text('  Jump: Home=max  End=min  Del=zero', color=(180, 180, 100))
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_text('  Arm: ', indent=4)
                    self._arm_var = dpg.add_radio_button(
                        ['Left', 'Right'],
                        default_value=0,
                        callback=self._on_arm_change,
                        horizontal=True,
                    )

                self._jnt_sliders = []
                self._jnt_labels  = []
                ranges_l = ctrl._jl_ranges
                for i in range(7):
                    lo_d = math.degrees(ranges_l[i][0])
                    hi_d = math.degrees(ranges_l[i][1])
                    cur  = math.degrees(float(ctrl.d.qpos[ctrl._jl_qadrs[i]]))
                    with dpg.group(horizontal=True):
                        dpg.add_text(f'  J{i+1}', indent=8)
                        tag = f'jnt_sl_{i}'
                        lbl = f'jnt_lbl_{i}'
                        dpg.add_slider_float(
                            tag=tag,
                            default_value=cur,
                            min_value=lo_d, max_value=hi_d,
                            width=220,
                            callback=lambda s, v, idx=i: self._on_joint(idx, v),
                            format='%.1f°',
                        )
                        dpg.add_text(f'{cur:+.1f}°', tag=lbl, indent=4)
                        self._jnt_sliders.append((tag, lo_d, hi_d))
                        self._jnt_labels.append(lbl)

                dpg.add_spacer(height=2)
                with dpg.group(horizontal=True):
                    dpg.add_button(label='All Zeros',
                                   callback=self._zero_all_joints, indent=8)
                    dpg.add_button(label='Reset Arm',
                                   callback=self._reset_arm)

            dpg.add_spacer(height=4)

            # ── Hand tab ────────────────────────────────────────────────
            with dpg.collapsing_header(label='Hand / Grip', default_open=True):
                dpg.add_text('  Z=toggle L grip   X=toggle R grip')
                dpg.add_separator()

                self._grip_sliders = {}
                self._grip_labels  = {}
                for side, label in [('l', 'Left  Grip'), ('r', 'Right Grip')]:
                    with dpg.group(horizontal=True):
                        dpg.add_text(f'  {label}', indent=4)
                        tag = f'grip_sl_{side}'
                        lbl = f'grip_lbl_{side}'
                        dpg.add_slider_float(
                            tag=tag,
                            default_value=0.0,
                            min_value=0.0, max_value=1.0,
                            width=200,
                            callback=lambda s, v, sd=side: self._on_grip(sd, v),
                            format='%.0f%%',
                        )
                        dpg.add_text('  0%', tag=lbl, indent=4)
                        self._grip_sliders[side] = tag
                        self._grip_labels[side]  = lbl
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='GRIP',
                                       callback=lambda s=side: self._set_grip(s, 1.0),
                                       indent=8, width=60)
                        dpg.add_button(label='OPEN',
                                       callback=lambda s=side: self._set_grip(s, 0.0))
                    dpg.add_spacer(height=4)

            dpg.add_spacer(height=4)

            # ── Status bar ──────────────────────────────────────────────
            dpg.add_separator()
            self._status_tag = dpg.add_text(
                'Status: initializing...', color=(150, 200, 150))

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_ik(self, user_data, value):
        """Slider moved → set IK target axis."""
        side, axis, lo, hi = user_data
        idx = {'x': 0, 'y': 1, 'z': 2}[axis[0]]
        try:
            tgt = (self._ctrl._ik_tgt_l_base if side == 'l'
                   else self._ctrl._ik_tgt_r_base)
            tgt[idx] = float(value)
        except Exception:
            pass

    def _on_joint(self, joint_idx, value_deg):
        """Joint slider moved → apply joint angle."""
        angle = math.radians(float(value_deg))
        arm   = 'l' if dpg.get_value(self._arm_var) == 0 else 'r'
        try:
            qadrs  = (self._ctrl._jl_qadrs if arm == 'l'
                      else self._ctrl._jr_qadrs)
            aids   = (self._ctrl._a_arm_l if arm == 'l'
                      else self._ctrl._a_arm_r)
            ranges = (self._ctrl._jl_ranges if arm == 'l'
                      else self._ctrl._jr_ranges)
            lo, hi = ranges[joint_idx]
            import numpy as np
            clamped = float(np.clip(angle, lo, hi))
            self._ctrl.d.qpos[qadrs[joint_idx]] = clamped
            self._ctrl.d.ctrl[aids[joint_idx]]  = clamped
            import mujoco
            mujoco.mj_forward(self._ctrl.m, self._ctrl.d)
        except Exception:
            pass

    def _on_arm_change(self, sender, value):
        """Arm radio button changed → refresh slider ranges and values."""
        arm    = 'l' if value == 0 else 'r'
        qadrs  = (self._ctrl._jl_qadrs if arm == 'l'
                  else self._ctrl._jr_qadrs)
        ranges = (self._ctrl._jl_ranges if arm == 'l'
                  else self._ctrl._jr_ranges)
        for i, (tag, _, _) in enumerate(self._jnt_sliders):
            lo_d = math.degrees(ranges[i][0])
            hi_d = math.degrees(ranges[i][1])
            cur  = math.degrees(float(self._ctrl.d.qpos[qadrs[i]]))
            dpg.configure_item(tag, min_value=lo_d, max_value=hi_d)
            dpg.set_value(tag, cur)
            self._jnt_sliders[i] = (tag, lo_d, hi_d)

    def _on_grip(self, side, value):
        try:
            if side == 'l':
                self._ctrl._grip_l = float(value)
            else:
                self._ctrl._grip_r = float(value)
        except Exception:
            pass

    def _set_grip(self, side, value):
        try:
            dpg.set_value(self._grip_sliders[side], value)
            self._on_grip(side, value)
        except Exception:
            pass

    def _zero_all_joints(self):
        arm    = 'l' if dpg.get_value(self._arm_var) == 0 else 'r'
        qadrs  = self._ctrl._jl_qadrs if arm == 'l' else self._ctrl._jr_qadrs
        aids   = self._ctrl._a_arm_l  if arm == 'l' else self._ctrl._a_arm_r
        ranges = self._ctrl._jl_ranges if arm == 'l' else self._ctrl._jr_ranges
        import numpy as np, mujoco
        for i in range(7):
            lo, hi = ranges[i]
            v = float(np.clip(0.0, lo, hi))
            self._ctrl.d.qpos[qadrs[i]] = v
            self._ctrl.d.ctrl[aids[i]]  = v
            tag, lo_d, hi_d = self._jnt_sliders[i]
            dpg.set_value(tag, math.degrees(v))
        mujoco.mj_forward(self._ctrl.m, self._ctrl.d)

    def _reset_arm(self):
        """Reset arm back to default qpos (same as mj_resetData default)."""
        import mujoco
        arm    = 'l' if dpg.get_value(self._arm_var) == 0 else 'r'
        qadrs  = self._ctrl._jl_qadrs if arm == 'l' else self._ctrl._jr_qadrs
        aids   = self._ctrl._a_arm_l  if arm == 'l' else self._ctrl._a_arm_r
        # default qpos for arm joints is typically 0
        import numpy as np
        ranges = self._ctrl._jl_ranges if arm == 'l' else self._ctrl._jr_ranges
        for i in range(7):
            lo, hi = ranges[i]
            v = float(np.clip(0.0, lo, hi))
            self._ctrl.d.qpos[qadrs[i]] = v
            self._ctrl.d.ctrl[aids[i]]  = v
            tag, lo_d, hi_d = self._jnt_sliders[i]
            dpg.set_value(tag, math.degrees(v))
        mujoco.mj_forward(self._ctrl.m, self._ctrl.d)

    # ── Poll: refresh display from simulation state ────────────────────────

    def _poll(self):
        ctrl = self._ctrl

        # IK target sliders + labels
        for key, tag in self._ik_sliders.items():
            side = key[0]
            idx  = {'x': 0, 'y': 1, 'z': 2}[key[2]]
            tgt  = ctrl._ik_tgt_l_base if side == 'l' else ctrl._ik_tgt_r_base
            val  = float(tgt[idx])
            # Only update slider if user is NOT dragging it (avoid fighting)
            if not dpg.is_item_active(tag):
                dpg.set_value(tag, val)
            dpg.set_value(self._ik_labels[key], f'{val:+.3f} m')

        # FK joint sliders + labels
        arm    = 'l' if dpg.get_value(self._arm_var) == 0 else 'r'
        qadrs  = ctrl._jl_qadrs if arm == 'l' else ctrl._jr_qadrs
        for i, (tag, lo_d, hi_d) in enumerate(self._jnt_sliders):
            cur = math.degrees(float(ctrl.d.qpos[qadrs[i]]))
            if not dpg.is_item_active(tag):
                dpg.set_value(tag, max(lo_d, min(hi_d, cur)))
            dpg.set_value(self._jnt_labels[i], f'{cur:+.1f}°')

        # Grip sliders + labels
        for side in ('l', 'r'):
            grip = ctrl._grip_l if side == 'l' else ctrl._grip_r
            tag  = self._grip_sliders[side]
            if not dpg.is_item_active(tag):
                dpg.set_value(tag, grip)
            dpg.set_value(self._grip_labels[side], f'{int(grip*100):3d}%')

        # Status bar
        el   = ctrl._ik_err_l * 1000.0
        er   = ctrl._ik_err_r * 1000.0
        mode = ctrl._mode.upper()
        freq = ctrl._freq_ema
        bx   = float(ctrl.d.qpos[ctrl._fj_qpos + 0])
        by   = float(ctrl.d.qpos[ctrl._fj_qpos + 1])
        dpg.set_value(
            self._status_tag,
            f'Mode:{mode}  IK-L:{el:.1f}mm  IK-R:{er:.1f}mm'
            f'  Freq:{freq:.0f}Hz  Base:({bx:.2f},{by:.2f})',
        )
