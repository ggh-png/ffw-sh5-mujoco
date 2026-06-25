"""Damped Least Squares (DLS) IK using MuJoCo's mj_jac().

Δθ = J^T (J J^T + λ²I)^{-1} e

mj_forward 는 함수 진입 전 1회만 호출된 것으로 가정.
각 이터레이션마다 관절 업데이트 후 forward 를 재계산한다.
"""
import numpy as np
import mujoco


def dls_ik(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    ee_id:       int,
    target_pos:  np.ndarray,
    dof_indices: list,
    qpos_addrs:  list,
    *,
    n_iter:   int   = 12,
    lam:      float = 0.01,
    max_dq:   float = 0.20,
    tol:      float = 1e-3,
) -> float:
    """단일 호출 당 최대 n_iter 번 반복. 진입 전 mj_forward 호출 필요.
    Returns final EE position error [m].
    """
    n    = len(dof_indices)
    jacp = np.zeros((3, model.nv))
    _jid_cache = _build_jid_cache(model, qpos_addrs)

    for _ in range(n_iter):
        ee_pos = data.xpos[ee_id].copy()
        err    = target_pos - ee_pos
        if np.linalg.norm(err) < tol:
            break

        jacp[:] = 0.0
        mujoco.mj_jac(model, data, jacp, None, ee_pos, ee_id)
        J   = jacp[:, dof_indices]
        JJT = J @ J.T + (lam ** 2) * np.eye(3)
        dq  = J.T @ np.linalg.solve(JJT, err)
        dq  = np.clip(dq, -max_dq, max_dq)

        for i in range(n):
            qadr    = qpos_addrs[i]
            lo, hi  = model.jnt_range[_jid_cache[i]]
            data.qpos[qadr] = float(np.clip(data.qpos[qadr] + dq[i], lo, hi))

        mujoco.mj_forward(model, data)

    return float(np.linalg.norm(target_pos - data.xpos[ee_id]))


def _build_jid_cache(model: mujoco.MjModel, qpos_addrs: list) -> list:
    cache = []
    for qadr in qpos_addrs:
        for j in range(model.njnt):
            if model.jnt_qposadr[j] == qadr:
                cache.append(j)
                break
    return cache
