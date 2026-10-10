"""Run four registration scenarios against a known ground-truth pose, print
the recovered-pose errors, and render the figures in docs/.

    python3 demo.py            # full run, writes docs/*.png and docs/icp.gif
    python3 demo.py --no-figs  # numbers only
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from icpkit import (add_noise, add_outliers, apply_T, crop_halfspace, estimate_normals, icp,
                    invert_T, make_scene, make_T, pose_error, register, so3_exp)

GT_SMALL = make_T(so3_exp(np.radians(25) * np.array([0.48, -0.6, 0.64])), np.array([0.20, -0.10, 0.15]))
GT_LARGE = make_T(so3_exp(np.radians(150) * np.array([0.0, 0.6, 0.8])), np.array([0.20, -0.10, 0.15]))


def make_pair(T_true, rng, n_dst=2500, n_src=2000):
    """dst and src are *independent* samples of the same surface -- no point in
    src has an exact twin in dst, as with two real scans."""
    dst, dn = make_scene(n_dst, rng)
    src_world, _ = make_scene(n_src, rng)
    src = apply_T(invert_T(T_true), src_world)  # so that T_true maps src onto dst
    return src, dst, dn


def row(name, res, T_true, secs):
    rot, tr = pose_error(res.T, T_true)
    print(f"  {name:<34} iters={res.iterations:>3}  rot err={rot:8.4f} deg  "
          f"trans err={tr:.5f}  rmse={res.rmse:.4f}  kept={res.inlier_fraction:4.0%}  {secs*1000:6.0f} ms")
    return res


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0


def main(figs=True):
    rng = np.random.default_rng(7)
    out = {}

    print("Scenario A: clean surfaces, 25 deg + translation, full overlap")
    src, dst, dn = make_pair(GT_SMALL, rng)
    a1, s = timed(icp, src, dst, method="point_to_point", max_iter=200); row("point-to-point", a1, GT_SMALL, s)
    a2, s = timed(icp, src, dst, method="point_to_plane", dst_normals=dn); row("point-to-plane (analytic normals)", a2, GT_SMALL, s)
    out["A"] = (src, dst, a1, a2)

    print("\nScenario B: Gaussian noise sigma=0.01 on both scans, PCA-estimated normals")
    src_b, dst_b = add_noise(src, 0.01, rng), add_noise(dst, 0.01, rng)
    dn_b = estimate_normals(dst_b, k=20)
    b1, s = timed(icp, src_b, dst_b, method="point_to_point", max_iter=200); row("point-to-point", b1, GT_SMALL, s)
    b2, s = timed(icp, src_b, dst_b, method="point_to_plane", dst_normals=dn_b); row("point-to-plane (estimated normals)", b2, GT_SMALL, s)

    print("\nScenario C: partial overlap (two half-space crops) + 15% outliers in src")
    dst_c, dn_c = crop_halfspace(dst, dn, (1, 0, 0), -0.45)                  # dst misses the far -x side
    src_world = apply_T(GT_SMALL, src)
    keep = src_world[:, 0] < 0.95                                           # src misses the +x sphere side
    src_c = add_outliers(src[keep], 0.15, rng)
    overlap = np.mean(src_world[keep][:, 0] > -0.45) * keep.sum() / len(src_c)
    print(f"  (fraction of src with a true counterpart in dst: {overlap:.0%})")
    c1, s = timed(icp, src_c, dst_c, method="point_to_plane", dst_normals=dn_c); row("plain point-to-plane", c1, GT_SMALL, s)
    c2, s = timed(icp, src_c, dst_c, method="point_to_plane", dst_normals=dn_c, trim=0.55); row("trimmed point-to-plane (keep 55%)", c2, GT_SMALL, s)
    out["C"] = (src_c, dst_c, c1, c2)

    print("\nScenario D: 150 deg initial rotation")
    src_d, dst_d, dn_d = make_pair(GT_LARGE, rng)
    d1, s = timed(icp, src_d, dst_d, method="point_to_plane", dst_normals=dn_d); row("point-to-plane from identity", d1, GT_LARGE, s)
    d2, s = timed(register, src_d, dst_d, dst_normals=dn_d); row("PCA multi-start + point-to-plane", d2, GT_LARGE, s)
    out["D"] = (src_d, dst_d, d1, d2)

    if figs:
        render_figures(out)
    return out


def render_figures(out):
    from icpkit.render import DST_COLOR, SRC_COLOR, convergence_chart, draw_clouds, icp_gif, panel_row
    os.makedirs("docs", exist_ok=True)

    src, dst, a1, a2 = out["A"]
    panel_row([
        draw_clouds([(dst, DST_COLOR), (src, SRC_COLOR)], title="A: initial pose"),
        draw_clouds([(dst, DST_COLOR), (apply_T(a1.T, src), SRC_COLOR)], title=f"point-to-point ({a1.iterations} it)"),
        draw_clouds([(dst, DST_COLOR), (apply_T(a2.T, src), SRC_COLOR)], title=f"point-to-plane ({a2.iterations} it)"),
    ]).save("docs/scenario_a.png")
    convergence_chart([("point-to-point", np.sqrt(a1.mse_history)),
                       ("point-to-plane", np.sqrt(a2.mse_history))],
                      title="Scenario A: pair RMSE per iteration").save("docs/convergence.png")
    rot_err = lambda r: [max(pose_error(T, GT_SMALL)[0], 1e-3) for T in r.T_history]
    convergence_chart([("point-to-point", rot_err(a1)), ("point-to-plane", rot_err(a2))],
                      title="Scenario A: rotation error vs ground truth (deg)").save("docs/rotation_error.png")
    icp_gif("docs/icp.gif", src, dst, a2.T_history)

    src_c, dst_c, c1, c2 = out["C"]
    panel_row([
        draw_clouds([(dst_c, DST_COLOR), (src_c, SRC_COLOR)], title="C: partial overlap + outliers"),
        draw_clouds([(dst_c, DST_COLOR), (apply_T(c1.T, src_c), SRC_COLOR)], title="plain ICP"),
        draw_clouds([(dst_c, DST_COLOR), (apply_T(c2.T, src_c), SRC_COLOR)], title="trimmed ICP"),
    ]).save("docs/scenario_c.png")

    src_d, dst_d, d1, d2 = out["D"]
    panel_row([
        draw_clouds([(dst_d, DST_COLOR), (src_d, SRC_COLOR)], title="D: 150 deg initial rotation"),
        draw_clouds([(dst_d, DST_COLOR), (apply_T(d1.T, src_d), SRC_COLOR)], title="ICP from identity"),
        draw_clouds([(dst_d, DST_COLOR), (apply_T(d2.T, src_d), SRC_COLOR)], title="PCA multi-start"),
    ]).save("docs/scenario_d.png")
    print("\nwrote docs/scenario_a.png docs/scenario_c.png docs/scenario_d.png docs/convergence.png docs/rotation_error.png docs/icp.gif")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    main(figs=not ap.parse_args().no_figs)
