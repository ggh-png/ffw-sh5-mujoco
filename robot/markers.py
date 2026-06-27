"""3D scene markers rendered in viewer.user_scn."""
import math
import numpy as np
import mujoco


def _mat_from_z_to_dir(d):
    """Rotation matrix mapping local Z=[0,0,1] to direction d."""
    d = np.asarray(d, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.eye(3)
    d = d / n
    if d[2] > 1.0 - 1e-8:
        return np.eye(3)
    if d[2] < -1.0 + 1e-8:
        return np.diag([1.0, -1.0, -1.0])
    v  = np.cross([0.0, 0.0, 1.0], d)
    s2 = np.dot(v, v) + 1e-30
    c  = d[2]
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + (vx @ vx) * (1.0 - c) / s2


def _add_geom(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        gtype,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos,  dtype=np.float64),
        np.asarray(mat,  dtype=np.float64).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _sphere(scn, pos, radius, rgba):
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
              [radius, 0, 0], pos, np.eye(3), rgba)


def _cylinder(scn, A, B, radius, rgba):
    diff = np.asarray(B, float) - np.asarray(A, float)
    L = np.linalg.norm(diff)
    if L < 1e-6:
        return
    mid = (np.asarray(A, float) + np.asarray(B, float)) * 0.5
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER,
              [radius, L * 0.5, 0], mid, _mat_from_z_to_dir(diff), rgba)


def render(scn, ctrl):
    """Rebuild user_scn each frame with IK markers + optional gizmo."""
    scn.ngeom = 0

    tgt_l = ctrl.ik_world_tgt_l
    tgt_r = ctrl.ik_world_tgt_r
    ee_l  = ctrl.ee_pos_l
    ee_r  = ctrl.ee_pos_r

    _sphere(scn, tgt_l, 0.026, [0.10, 0.90, 0.10, 0.85])
    _sphere(scn, tgt_r, 0.026, [0.10, 0.80, 0.90, 0.85])
    _sphere(scn, ee_l,  0.018, [1.00, 0.50, 0.00, 0.70])
    _sphere(scn, ee_r,  0.018, [1.00, 1.00, 0.00, 0.70])

    _cylinder(scn, ee_l, tgt_l, 0.004, [0.10, 0.90, 0.10, 0.45])
    _cylinder(scn, ee_r, tgt_r, 0.004, [0.10, 0.80, 0.90, 0.45])

    # 태스크 활성 시: 파지점 / 접근점 마커
    task = getattr(ctrl, 'task', None)
    if task is not None and task.is_active():
        vis = task.vis_target
        if vis is not None:
            _sphere(scn, vis, 0.022, [1.00, 0.85, 0.00, 0.90])   # 노란색: 현재 목표
        if task.grasp_pos is not None:
            _sphere(scn, task.grasp_pos, 0.015, [1.00, 0.40, 0.00, 0.70])  # 주황: 파지점
        if task.pre_grasp_pos is not None:
            _sphere(scn, task.pre_grasp_pos, 0.012, [0.80, 0.80, 0.00, 0.60])  # 연노랑: 접근점

    if ctrl.show_gizmo:
        p  = ctrl.base_world_pos.copy()
        yaw = ctrl.base_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        L = 0.35
        ax = np.array([ c,  s, 0.0])
        ay = np.array([-s,  c, 0.0])
        az = np.array([ 0.0, 0.0, 1.0])
        _cylinder(scn, p, p + ax * L, 0.007, [1.0, 0.2, 0.2, 0.9])
        _cylinder(scn, p, p + ay * L, 0.007, [0.2, 1.0, 0.2, 0.9])
        _cylinder(scn, p, p + az * L, 0.007, [0.2, 0.2, 1.0, 0.9])
