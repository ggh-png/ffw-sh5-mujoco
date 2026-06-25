"""Bounded-Variable Least Squares (BVLS) IK — convex QP formulation.

Solves each iteration as a proper constrained QP:
    min  ||J Δθ - e||² + λ||Δθ||²
    s.t. θ_lo ≤ θ + Δθ ≤ θ_hi   (hard joint-limit constraints)

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
    n_iter:  int   = 15,
    lam:     float = 0.005,   # Tikhonov regularisation
    max_dq:  float = 0.30,    # per-step joint-velocity limit [rad]
    tol:     float = 1e-3,    # position error tolerance [m]
) -> float:
    """Run ≤n_iter QP iterations.  Returns final ||error|| [m].

    Call mj_forward() before the first invocation so Jacobians are valid.
    """
    n     = len(dof_indices)
    jacp  = np.zeros((3, model.nv))
    jids  = _jid_cache(model, qpos_addrs)

    lo = np.array([model.jnt_range[jids[i]][0] for i in range(n)])
    hi = np.array([model.jnt_range[jids[i]][1] for i in range(n)])
    sq_lam = np.sqrt(lam)

    for _ in range(n_iter):
        ee_pos = data.xpos[ee_id].copy()
        err    = target_pos - ee_pos
        if np.linalg.norm(err) < tol:
            break

        jacp[:] = 0.0
        mujoco.mj_jac(model, data, jacp, None, ee_pos, ee_id)
        J = jacp[:, dof_indices]           # (3, n)

        # Augment system: [J; sqrt(λ)·I] dq = [e; 0]
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
            JJT = J @ J.T + lam * np.eye(3)
            dq  = np.clip(J.T @ np.linalg.solve(JJT, err), dq_lo, dq_hi)

        for i, qadr in enumerate(qpos_addrs):
            data.qpos[qadr] = float(np.clip(data.qpos[qadr] + dq[i],
                                            lo[i], hi[i]))

        mujoco.mj_forward(model, data)

    return float(np.linalg.norm(target_pos - data.xpos[ee_id]))


def _jid_cache(model, qpos_addrs):
    """Map qpos address → joint index."""
    cache = []
    for qadr in qpos_addrs:
        for j in range(model.njnt):
            if model.jnt_qposadr[j] == qadr:
                cache.append(j)
                break
    return cache
