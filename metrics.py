"""
metrics.py

Unified metrics for belief system dynamics simulation.

Assumed trajectory shape:
    traj: (T, n, m)
"""

import numpy as np


# =========================================================
# 🔹 基础工具
# =========================================================

def _to_numpy(traj):
    return np.array(traj)


def _final(traj):
    return traj[-1]


# =========================================================
# 🔹 1. MEAN (群体方向)
# =========================================================

def mean_trajectory(traj):
    """
    Return mean belief at each timestep.
    shape: (T, m)
    """
    traj = _to_numpy(traj)
    return traj.mean(axis=1)


def final_mean(traj):
    return mean_trajectory(traj)[-1]


# =========================================================
# 🔹 2. VARIANCE (一致性)
# =========================================================

def variance_trajectory(traj):
    """
    Return disagreement (variance) over time.
    shape: (T,)
    """
    traj = _to_numpy(traj)
    mean = traj.mean(axis=1, keepdims=True)
    var = np.mean(np.linalg.norm(traj - mean, axis=2)**2, axis=1)
    return var


def final_variance(traj):
    return variance_trajectory(traj)[-1]


# =========================================================
# 🔹 3. ALIGNMENT (KOL影响力)
# =========================================================

def alignment_with_kol(traj, kol_index):
    """
    Measure average distance to KOL at each timestep.
    shape: (T,)
    """
    traj = _to_numpy(traj)
    x_kol = traj[:, kol_index, :]   # (T, m)

    diff = traj - x_kol[:, None, :]
    align = np.mean(np.linalg.norm(diff, axis=2), axis=1)

    return align


def final_alignment(traj, kol_index):
    return alignment_with_kol(traj, kol_index)[-1]

# =========================================================
# 🔹 6. 高层接口（推荐使用）
# =========================================================

def compute_all_metrics(traj, kol_index=None):
    """
    Unified metrics output.
    """
    traj = _to_numpy(traj)

    result = {
        "mean_traj": mean_trajectory(traj),
        "variance_traj": variance_trajectory(traj),
        "final_mean": final_mean(traj),
        "final_variance": final_variance(traj)
    }

    if kol_index is not None:
        result["alignment_traj"] = alignment_with_kol(traj, kol_index)
        result["final_alignment"] = result["alignment_traj"][-1]

    return result