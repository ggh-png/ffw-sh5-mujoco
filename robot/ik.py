"""Bounded-Variable Least Squares (BVLS) IK — convex QP formulation.

3DOF (위치만):
    min  ||Jp Δθ - e_pos||² + λ||Δθ||²
    s.t. θ_lo ≤ θ + Δθ ≤ θ_hi

6DOF (위치 + 자세, target_quat 제공 시):
    min  ||[Jp; Jr·w] Δθ - [e_pos; e_ori·w]||² + λ||Δθ||²
    s.t. θ_lo ≤ θ + Δθ ≤ θ_hi

Uses scipy.optimize.lsq_linear (BVLS algorithm) which guarantees the
global optimum of each linearised sub-problem — unlike DLS+clamp which
violates joint limits when clamping is needed.
"""
import numpy as np
import mujoco
from scipy.optimize import lsq_linear


def qp_ik(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    ee_id:       int,
    target_pos:  np.ndarray,
    dof_indices: list,
    qpos_addrs:  list,
    *,
    target_quat:   'np.ndarray | None' = None,  # [w,x,y,z] world-frame; None = pos-only
    palm_body_ids: 'list | None' = None,         # bodies whose mean pos = palm center
    rot_weight:  float = 0.5,   # orientation rows weight relative to position
    n_iter:  int   = 15,
    lam:     float = 0.005,     # Tikhonov regularisation
    max_dq:  float = 0.30,      # per-step joint-velocity limit [rad]
    tol:     float = 1e-3,      # position error tolerance [m]
) -> float:
    """Run ≤n_iter QP iterations.  Returns final position ||error|| [m].

    Call mj_forward() before the first invocation so Jacobians are valid.
    When target_quat is provided, 6-DOF IK is used (position + orientation).
    When palm_body_ids is provided, the IK targets the mean position of those
    bodies (palm grasp center) instead of ee_id's body origin.  The rotation
    Jacobian still uses ee_id — palm bodies are children of ee_id so the
    translation Jacobian at that point is exact for arm DOFs.
    """
    n      = len(dof_indices)
    jacp   = np.zeros((3, model.nv))
    jacr   = np.zeros((3, model.nv)) if target_quat is not None else None
    jids   = _jid_cache(model, qpos_addrs)
    use6   = target_quat is not None

    lo = np.array([model.jnt_range[jids[i]][0] for i in range(n)])
    hi = np.array([model.jnt_range[jids[i]][1] for i in range(n)])
    sq_lam = np.sqrt(lam)

    for _ in range(n_iter):
        # Palm center or EE body origin as the position reference
        if palm_body_ids:
            ee_pos = np.mean([data.xpos[b] for b in palm_body_ids], axis=0)
        else:
            ee_pos = data.xpos[ee_id].copy()
        pos_err = target_pos - ee_pos
        if np.linalg.norm(pos_err) < tol:
            break

        jacp[:] = 0.0
        if use6:
            jacr[:] = 0.0
        # point=ee_pos: translation Jacobian at palm center (arm DOFs are exact);
        # body=ee_id:   rotation Jacobian for EE orientation (unchanged)
        mujoco.mj_jac(model, data, jacp, jacr, ee_pos, ee_id)
        Jp = jacp[:, dof_indices]   # (3, n)

        if use6:
            # Orientation error: skew-symmetric extraction of R_err = R_tgt @ R_cur.T
            R_cur     = data.xmat[ee_id].reshape(3, 3)
            R_tgt_buf = np.zeros(9)
            mujoco.mju_quat2Mat(R_tgt_buf, target_quat)
            R_tgt  = R_tgt_buf.reshape(3, 3)
            R_err  = R_tgt @ R_cur.T
            ori_err = np.array([R_err[2, 1] - R_err[1, 2],
                                R_err[0, 2] - R_err[2, 0],
                                R_err[1, 0] - R_err[0, 1]]) * 0.5
            Jr  = jacr[:, dof_indices] * rot_weight   # (3, n)
            J   = np.vstack([Jp, Jr])                 # (6, n)
            err = np.concatenate([pos_err, ori_err * rot_weight])
        else:
            J   = Jp
            err = pos_err

        ne = J.shape[0]
        J_aug = np.vstack([J, sq_lam * np.eye(n)])
        e_aug = np.concatenate([err, np.zeros(n)])

        # Per-step bound: clamp to [lo-θ, hi-θ] ∩ [-max_dq, max_dq]
        q = np.array([data.qpos[a] for a in qpos_addrs])
        dq_lo = np.maximum(lo - q, -max_dq)
        dq_hi = np.minimum(hi - q,  max_dq)

        try:
            res = lsq_linear(J_aug, e_aug,
                             bounds=(dq_lo, dq_hi),
                             method='bvls', max_iter=40, tol=1e-7)
            dq = res.x
        except Exception:
            # Fallback: unconstrained DLS + clamp
            JJT = J @ J.T + lam * np.eye(ne)
            dq  = np.clip(J.T @ np.linalg.solve(JJT, err), dq_lo, dq_hi)

        for i, qadr in enumerate(qpos_addrs):
            data.qpos[qadr] = float(np.clip(data.qpos[qadr] + dq[i],
                                            lo[i], hi[i]))

        mujoco.mj_forward(model, data)

    if palm_body_ids:
        final_pos = np.mean([data.xpos[b] for b in palm_body_ids], axis=0)
    else:
        final_pos = data.xpos[ee_id].copy()
    return float(np.linalg.norm(target_pos - final_pos))


def _jid_cache(model, qpos_addrs):
    """Map qpos address → joint index."""
    cache = []
    for qadr in qpos_addrs:
        for j in range(model.njnt):
            if model.jnt_qposadr[j] == qadr:
                cache.append(j)
                break
    return cache
