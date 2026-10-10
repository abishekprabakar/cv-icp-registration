"""Pillow-only rendering: orthographic point-cloud views, a log-scale
convergence chart, and an animated GIF of ICP iterations."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .se3 import apply_T, so3_exp

BG = (250, 250, 247)
INK = (40, 40, 40)
GRID = (222, 222, 216)
DST_COLOR = (38, 110, 190)    # blue: target
SRC_COLOR = (220, 90, 40)     # orange: moving source
SERIES = [(220, 90, 40), (38, 110, 190), (40, 150, 90), (150, 80, 170)]

VIEW = so3_exp(np.array([-1.0, 0.0, 0.0]) * np.radians(60)) @ so3_exp(np.array([0, 0, 1.0]) * np.radians(30))


def _project(P, size, center, scale):
    V = (P - center) @ VIEW.T
    x = size[0] / 2 + V[:, 0] * scale
    y = size[1] / 2 - V[:, 1] * scale
    return x, y, V[:, 2]


def draw_clouds(clouds, size=(420, 360), title=None, center=None, scale=None, radius=1.6):
    """clouds: list of (points, rgb).  Points from all clouds are depth-sorted
    together so the overlap reads correctly."""
    allP = np.vstack([c for c, _ in clouds])
    center = allP.mean(0) if center is None else center
    if scale is None:
        ext = np.abs((allP - center) @ VIEW.T)[:, :2].max()
        scale = 0.42 * min(size) / ext
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    xs, ys, zs, cols = [], [], [], []
    for P, rgb in clouds:
        x, y, z = _project(P, size, center, scale)
        xs.append(x); ys.append(y); zs.append(z); cols += [rgb] * len(P)
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    zn = (z - z.min()) / max(np.ptp(z), 1e-9)
    for i in np.argsort(z):  # far to near
        shade = 0.55 + 0.45 * zn[i]
        c = tuple(int(BG[k] + (cols[i][k] - BG[k]) * shade) for k in range(3))
        d.ellipse([x[i] - radius, y[i] - radius, x[i] + radius, y[i] + radius], fill=c)
    if title:
        d.text((10, 8), title, fill=INK)
    return img


def panel_row(images, gap=8):
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    h = max(i.height for i in images)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def convergence_chart(series, size=(640, 360), title="RMSE vs iteration (log scale)"):
    """series: list of (label, values).  Draws log10(values) against iteration."""
    W, H = size
    L, R, T, B = 64, 20, 34, 40
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    vals = np.concatenate([np.asarray(v, float) for _, v in series])
    vals = vals[vals > 0]
    lo, hi = np.floor(np.log10(vals.min())), np.ceil(np.log10(vals.max()))
    if hi == lo:
        hi += 1
    n_max = max(len(v) for _, v in series)
    X = lambda i: L + (W - L - R) * (i / max(n_max - 1, 1))
    Y = lambda v: T + (H - T - B) * (hi - np.log10(v)) / (hi - lo)
    for e in range(int(lo), int(hi) + 1):
        y = Y(10.0**e)
        d.line([(L, y), (W - R, y)], fill=GRID)
        d.text((8, y - 6), f"1e{e}", fill=INK)
    d.line([(L, T), (L, H - B), (W - R, H - B)], fill=INK)
    for k in range(0, n_max, max(1, n_max // 8)):
        d.text((X(k) - 4, H - B + 6), str(k + 1), fill=INK)
    d.text((W // 2 - 30, H - 18), "iteration", fill=INK)
    d.text((L, 10), title, fill=INK)
    for s, (label, v) in enumerate(series):
        col = SERIES[s % len(SERIES)]
        pts = [(X(i), Y(max(val, 10.0**lo))) for i, val in enumerate(v)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
        d.rectangle([W - R - 190, T + 8 + 18 * s, W - R - 178, T + 20 + 18 * s], fill=col)
        d.text((W - R - 172, T + 7 + 18 * s), label, fill=INK)
    return img


def icp_gif(path, src, dst, T_history, size=(420, 360), every=1, ms=180):
    center = dst.mean(0)
    ext = np.abs((dst - center) @ VIEW.T)[:, :2].max()
    scale = 0.36 * min(size) / ext
    frames = []
    hist = T_history[::every]
    if hist[-1] is not T_history[-1]:
        hist = hist + [T_history[-1]]
    for k, T in enumerate(hist):
        frames.append(draw_clouds([(dst, DST_COLOR), (apply_T(T, src), SRC_COLOR)], size=size,
                                  center=center, scale=scale, radius=1.3,
                                  title=f"ICP iteration {min(k * every, len(T_history) - 1)}"))
    frames += [frames[-1]] * 6
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
