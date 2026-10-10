"""Procedural point clouds with analytic normals, plus neighbour search and
PCA normal estimation.

Everything is numpy-only so the repo runs anywhere.  Nearest-neighbour search
is exact brute force done in chunks: for the few-thousand-point clouds used
here that is a few milliseconds per query batch and needs no k-d tree.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# surface samplers (area-uniform) -- each returns (points, unit normals)
# ---------------------------------------------------------------------------
def sample_sphere(n, rng, center=(0, 0, 0), radius=1.0):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.asarray(center) + radius * d, d


def sample_torus(n, rng, R=1.0, r=0.3, center=(0, 0, 0)):
    """Area-uniform torus samples.  The area element is r (R + r cos v) du dv,
    so v is drawn by rejection against (R + r cos v) / (R + r)."""
    out_v = np.empty(0)
    while out_v.size < n:
        v = rng.uniform(0, 2 * np.pi, size=2 * n)
        keep = rng.uniform(0, 1, size=2 * n) < (R + r * np.cos(v)) / (R + r)
        out_v = np.concatenate([out_v, v[keep]])
    v = out_v[:n]
    u = rng.uniform(0, 2 * np.pi, size=n)
    nrm = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], axis=1)
    ring = np.stack([R * np.cos(u), R * np.sin(u), np.zeros(n)], axis=1)
    return np.asarray(center) + ring + r * nrm, nrm


def sample_box(n, rng, center=(0, 0, 0), half=(0.5, 0.5, 0.5)):
    """Uniform samples over the six faces of an axis-aligned box."""
    hx, hy, hz = half
    areas = np.array([hy * hz, hy * hz, hx * hz, hx * hz, hx * hy, hx * hy])
    face = rng.choice(6, size=n, p=areas / areas.sum())
    P = rng.uniform(-1, 1, size=(n, 3)) * np.asarray(half)
    N = np.zeros((n, 3))
    axis = face // 2
    sign = np.where(face % 2 == 0, 1.0, -1.0)
    P[np.arange(n), axis] = sign * np.asarray(half)[axis]
    N[np.arange(n), axis] = sign
    return np.asarray(center) + P, N


def make_scene(n, rng):
    """An asymmetric test object: torus + off-axis sphere + box in the hole.

    The torus alone is rotationally symmetric about z (ICP would have a
    continuum of equally good answers); the sphere and box break that.
    Sample counts are split in proportion to surface area so density is
    uniform over the whole object.
    """
    parts = [
        ("torus", 4 * np.pi**2 * 1.0 * 0.3),
        ("sphere", 4 * np.pi * 0.35**2),
        ("box", 8 * (0.25 * 0.15 + 0.15 * 0.4 + 0.25 * 0.4)),
    ]
    area = np.array([a for _, a in parts])
    counts = np.floor(n * area / area.sum()).astype(int)
    counts[0] += n - counts.sum()
    P1, N1 = sample_torus(counts[0], rng, R=1.0, r=0.3)
    P2, N2 = sample_sphere(counts[1], rng, center=(1.3, 0.0, 0.7), radius=0.35)
    P3, N3 = sample_box(counts[2], rng, center=(0.1, 0.0, 0.0), half=(0.25, 0.15, 0.4))
    return np.vstack([P1, P2, P3]), np.vstack([N1, N2, N3])


# ---------------------------------------------------------------------------
# corruptions
# ---------------------------------------------------------------------------
def add_noise(P, sigma, rng):
    return P + rng.normal(scale=sigma, size=P.shape)


def add_outliers(P, frac, rng, pad=0.3):
    """Append frac*len(P) points uniform in the padded bounding box."""
    k = int(round(frac * len(P)))
    lo, hi = P.min(0) - pad, P.max(0) + pad
    return np.vstack([P, rng.uniform(lo, hi, size=(k, 3))])


def crop_halfspace(P, N, normal, offset):
    """Keep points with dot(p, normal) > offset (simulates a partial scan)."""
    keep = P @ np.asarray(normal, float) > offset
    return P[keep], (None if N is None else N[keep])


# ---------------------------------------------------------------------------
# neighbour search + normals
# ---------------------------------------------------------------------------
def nearest_neighbors(src, dst, chunk=1024):
    """For every row of src, the index of and distance to its nearest row in dst."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    d2_dst = (dst**2).sum(1)
    idx = np.empty(len(src), dtype=np.int64)
    dist = np.empty(len(src))
    for s in range(0, len(src), chunk):
        A = src[s:s + chunk]
        d2 = (A**2).sum(1)[:, None] - 2.0 * A @ dst.T + d2_dst[None, :]
        j = np.argmin(d2, axis=1)
        idx[s:s + chunk] = j
        # recompute the winning distance exactly (the expansion above can lose
        # a few digits through cancellation)
        dist[s:s + chunk] = np.linalg.norm(A - dst[j], axis=1)
    return idx, dist


def knn(P, k, chunk=512):
    """Indices of the k nearest neighbours of each point (including itself)."""
    P = np.asarray(P, float)
    sq = (P**2).sum(1)
    out = np.empty((len(P), k), dtype=np.int64)
    for s in range(0, len(P), chunk):
        A = P[s:s + chunk]
        d2 = sq[s:s + chunk, None] - 2.0 * A @ P.T + sq[None, :]
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(d2, part, axis=1).argsort(axis=1)
        out[s:s + chunk] = np.take_along_axis(part, order, axis=1)
    return out


def estimate_normals(P, k=16, viewpoint=None):
    """Unit normals from the smallest-eigenvalue eigenvector of each point's
    k-neighbourhood covariance (Hoppe et al. 1992).

    PCA cannot tell n from -n.  If a viewpoint is given, normals are flipped
    to face it; otherwise they are flipped to point away from the cloud's
    centroid, which is right for the convex-ish objects used here and is
    irrelevant for point-to-plane ICP (the residual n.(p-q) only changes sign).
    """
    P = np.asarray(P, float)
    nb = knn(P, k)
    X = P[nb] - P[nb].mean(axis=1, keepdims=True)  # (N, k, 3)
    C = np.einsum("nki,nkj->nij", X, X)
    _, vecs = np.linalg.eigh(C)
    Nrm = vecs[:, :, 0]
    ref = P.mean(0) if viewpoint is None else np.asarray(viewpoint, float)
    flip = ((P - ref) * Nrm).sum(1) < 0
    if viewpoint is not None:
        flip = ~flip
    Nrm[flip] *= -1
    return Nrm
