"""Rigid-body geometry: SO(3)/SE(3) maps and closed-form absolute orientation.

Three independent solutions to the absolute-orientation problem live here:

* ``kabsch``           -- SVD of the weighted cross-covariance (Kabsch 1976,
                          Arun-Huang-Blostein 1987) with the det(R)=+1 fix.
* ``horn_quaternion``  -- Horn 1987: the optimal rotation is the unit
                          eigenvector of a symmetric 4x4 matrix built from the
                          same cross-covariance.  Used in the tests as an
                          independent check on ``kabsch``.
* ``umeyama``          -- Umeyama 1991 similarity (rotation + translation +
                          isotropic scale).
"""
from __future__ import annotations

import numpy as np


def skew(w: np.ndarray) -> np.ndarray:
    """3x3 matrix [w]_x such that [w]_x v = w x v."""
    x, y, z = w
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation by angle |w| about axis w/|w|."""
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w)
    K = skew(w)
    if theta < 1e-8:
        # second-order Taylor expansion keeps R orthonormal to ~1e-16
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + (np.sin(theta) / theta) * K + ((1 - np.cos(theta)) / theta**2) * K @ K


def so3_log(R: np.ndarray) -> np.ndarray:
    """Inverse of so3_exp; returns the rotation vector (angle in [0, pi])."""
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if np.pi - theta < 1e-6:
        # near pi: axis from the dominant column of (R + I)/2 = a a^T
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        a = B[:, i] / np.sqrt(B[i, i])
        return a * theta
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * theta / (2.0 * np.sin(theta))


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_T(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N, 3) array of points."""
    return P @ T[:3, :3].T + T[:3, 3]


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rotation_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    c = np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def pose_error(T_est: np.ndarray, T_true: np.ndarray) -> tuple[float, float]:
    """(rotation error in degrees, translation error in scene units)."""
    return (
        rotation_error_deg(T_est[:3, :3], T_true[:3, :3]),
        float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])),
    )


def random_rotation(rng: np.random.Generator, max_angle_deg: float | None = None) -> np.ndarray:
    """Uniform random rotation, or a random axis with a fixed angle."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    if max_angle_deg is None:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        return quat_to_R(q)
    return so3_exp(axis * np.radians(max_angle_deg))


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _weighted_centroids(P, Q, w):
    if w is None:
        w = np.ones(len(P))
    w = np.asarray(w, dtype=float)
    ws = w.sum()
    if ws <= 0:
        raise ValueError("weights must have a positive sum")
    mp = (w[:, None] * P).sum(0) / ws
    mq = (w[:, None] * Q).sum(0) / ws
    return w, ws, mp, mq


def kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rigid (R, t) minimising sum_i w_i |R p_i + t - q_i|^2.

    Takes paired (N, 3) arrays.  The sign correction D = diag(1, 1, det(V U^T))
    guarantees a proper rotation even when the best orthogonal fit would be a
    reflection (degenerate or mirrored data).
    """
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P and Q must both be (N, 3)")
    if len(P) < 3:
        raise ValueError("need at least 3 correspondences")
    w, _, mp, mq = _weighted_centroids(P, Q, w)
    H = ((P - mp) * w[:, None]).T @ (Q - mq)  # 3x3 cross-covariance
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mq - R @ mp
    return R, t


def horn_quaternion(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Horn's closed form: the optimal unit quaternion is the eigenvector of
    the largest eigenvalue of a 4x4 symmetric matrix N built from H."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    w, _, mp, mq = _weighted_centroids(P, Q, w)
    S = ((P - mp) * w[:, None]).T @ (Q - mq)
    Sxx, Sxy, Sxz = S[0]
    Syx, Syy, Syz = S[1]
    Szx, Szy, Szz = S[2]
    N = np.array([
        [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
        [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
        [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
        [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
    ])
    vals, vecs = np.linalg.eigh(N)
    q = vecs[:, np.argmax(vals)]
    R = quat_to_R(q)
    return R, mq - R @ mp


def umeyama(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising sum |s R p_i + t - q_i|^2 (Umeyama 1991)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    n = len(P)
    mp, mq = P.mean(0), Q.mean(0)
    Pc, Qc = P - mp, Q - mq
    var_p = (Pc**2).sum() / n
    Sigma = Qc.T @ Pc / n
    U, Dv, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(Dv) @ S) / var_p)
    t = mq - s * R @ mp
    return s, R, t
