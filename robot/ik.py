"""Damped Least Squares (DLS) IK using MuJoCo's mj_jac().

Δθ = J^T (J J^T + λ²I)^{-1} e
"""
import numpy as np
import mujoco


def dls_ik(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    ee_id:       int,
    target_pos:  np.ndarray,
    dof_indices: list,   # velocity DOF addresses (model.jnt_dofadr)
    qpos_addrs:  list,   # qpos addresses         (model.jnt_qposadr)
    *,
    n_iter:   int   = 100,
    lam:      float = 0.01,
    max_dq:   float = 0.20,
    tol:      float = 1e-3,
) -> float:
    """Returns final EE position error [m]."""
    n = len(dof_indices)
    jacp = np.zeros((3, model.nv))

    for _ in range(n_iter):
        mujoco.mj_forward(model, data)
        ee_pos = data.xpos[ee_id].copy()
        err    = target_pos - ee_pos
        if np.linalg.norm(err) < tol:
            break

        jacp[:] = 0.0
        mujoco.mj_jac(model, data, jacp, None, ee_pos, ee_id)
        J   = jacp[:, dof_indices]             # 3 × n
        JJT = J @ J.T + (lam ** 2) * np.eye(3)
        dq  = J.T @ np.linalg.solve(JJT, err)
        dq  = np.clip(dq, -max_dq, max_dq)

        for i in range(n):
            qadr     = qpos_addrs[i]
            jid      = _find_jid(model, qadr)
            lo, hi   = model.jnt_range[jid]
            data.qpos[qadr] = float(np.clip(data.qpos[qadr] + dq[i], lo, hi))

    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target_pos - data.xpos[ee_id]))


def _find_jid(model: mujoco.MjModel, qadr: int) -> int:
    for j in range(model.njnt):
        if model.jnt_qposadr[j] == qadr:
            return j
    raise ValueError(f'No joint at qposadr={qadr}')
