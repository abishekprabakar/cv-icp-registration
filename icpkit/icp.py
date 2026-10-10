"""Iterative Closest Point: point-to-point, point-to-plane, trimmed, and a
PCA-seeded multi-start for large initial rotations.

Conventions: we estimate T (4x4) such that apply_T(T, src) lands on dst.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clouds import nearest_neighbors
from .se3 import apply_T, kabsch, make_T, so3_exp


@dataclass
class ICPResult:
    T: np.ndarray
    iterations: int
    converged: bool
    mse_history: list = field(default_factory=list)    # mean sq. distance of kept pairs, measured at each correspondence step
    T_history: list = field(default_factory=list)      # pose after each update (for animation)
    inlier_fraction: float = 1.0

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.mse_history[-1])) if self.mse_history else float("nan")


def _select_pairs(dist, trim, max_dist):
    """Trimmed ICP (Chetverikov et al. 2002): keep the ceil(trim*N) closest
    pairs, then drop anything beyond max_dist."""
    n = len(dist)
    keep_n = max(3, int(np.ceil(trim * n)))
    if keep_n >= n:
        sel = np.arange(n)
    else:
        sel = np.argpartition(dist, keep_n - 1)[:keep_n]
    if max_dist is not None:
        sel = sel[dist[sel] <= max_dist]
    return sel


def _point_to_plane_step(p, q, n):
    """One Gauss-Newton step of min sum (n_i . (R p_i + t - q_i))^2.

    Linearising R ~ I + [w]_x gives the residual
        r_i = n_i.(p_i - q_i) + w.(p_i x n_i) + t.n_i
    so each pair contributes a row J_i = [p_i x n_i, n_i] (Chen & Medioni 1991,
    Low 2004).  The step is mapped back onto SO(3) with the exponential map,
    so the pose stays exactly rigid.
    """
    J = np.hstack([np.cross(p, n), n])          # (M, 6)
    r = ((p - q) * n).sum(1)                    # (M,)
    A = J.T @ J
    b = -J.T @ r
    # tiny Tikhonov term: keeps A invertible when the overlap is a single
    # plane (which leaves in-plane sliding unconstrained)
    x = np.linalg.solve(A + 1e-12 * np.trace(A) * np.eye(6), b)
    return make_T(so3_exp(x[:3]), x[3:])


def icp(
    src,
    dst,
    *,
    method="point_to_point",
    dst_normals=None,
    init=None,
    max_iter=60,
    tol=1e-10,
    trim=1.0,
    max_dist=None,
):
    """Register src onto dst.

    method      "point_to_point" (Besl & McKay 1992) or "point_to_plane".
    trim        fraction of closest pairs kept each iteration (1.0 = plain ICP).
    max_dist    optional hard gate on pair distance.
    tol         stop when the relative drop in pair MSE is below tol, or the
                incremental pose update is below ~1e-9.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if method not in ("point_to_point", "point_to_plane"):
        raise ValueError(f"unknown method {method!r}")
    if method == "point_to_plane" and dst_normals is None:
        raise ValueError("point_to_plane needs dst_normals")
    if not 0 < trim <= 1:
        raise ValueError("trim must be in (0, 1]")

    T = np.eye(4) if init is None else np.array(init, float)
    res = ICPResult(T=T, iterations=0, converged=False, T_history=[T.copy()])
    prev = None
    for it in range(1, max_iter + 1):
        cur = apply_T(T, src)
        idx, dist = nearest_neighbors(cur, dst)
        sel = _select_pairs(dist, trim, max_dist)
        if len(sel) < 6:
            break
        mse = float(np.mean(dist[sel] ** 2))
        res.mse_history.append(mse)
        res.inlier_fraction = len(sel) / len(src)

        p, q = cur[sel], dst[idx[sel]]
        if method == "point_to_point":
            R, t = kabsch(p, q)
            dT = make_T(R, t)
        else:
            dT = _point_to_plane_step(p, q, dst_normals[idx[sel]])
        T = dT @ T
        res.T_history.append(T.copy())
        res.iterations = it

        step = np.linalg.norm(dT[:3, :3] - np.eye(3)) + np.linalg.norm(dT[:3, 3])
        if prev is not None and (prev - mse) <= tol * max(prev, 1e-30) and step < 1e-6:
            res.converged = True
            break
        if step < 1e-9:
            res.converged = True
            break
        prev = mse
    res.T = T
    return res


def pca_frame(P):
    """Centroid and principal axes (columns, descending variance) of a cloud."""
    c = P.mean(0)
    _, vecs = np.linalg.eigh(np.cov((P - c).T))
    return c, vecs[:, ::-1]


def pca_candidates(src, dst):
    """The 4 proper rotations that map src's principal frame onto dst's.

    Eigenvectors are only defined up to sign; of the 8 sign patterns, the 4
    with det = +1 are rigid.  One of them is near the truth whenever the
    principal variances are distinct, regardless of the initial rotation.
    """
    cs, Es = pca_frame(src)
    cd, Ed = pca_frame(dst)
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            S = np.diag([sx, sy, sx * sy * np.linalg.det(Es) * np.linalg.det(Ed)])
            R = Ed @ S @ Es.T
            out.append(make_T(R, cd - R @ cs))
    return out


def register(src, dst, *, dst_normals=None, method=None, trim=1.0, max_iter=60, coarse_iter=15):
    """Global-ish registration: run a short ICP from identity and from each
    PCA candidate, keep the best by trimmed MSE, then refine it fully."""
    method = method or ("point_to_plane" if dst_normals is not None else "point_to_point")
    seeds = [np.eye(4)] + pca_candidates(src, dst)
    trials = [
        icp(src, dst, method=method, dst_normals=dst_normals, init=s, max_iter=coarse_iter, trim=trim)
        for s in seeds
    ]
    best = min(trials, key=lambda r: r.mse_history[-1])
    return icp(src, dst, method=method, dst_normals=dst_normals, init=best.T, max_iter=max_iter, trim=trim)
