"""ICP against known ground-truth transforms."""
import unittest

import numpy as np

from icpkit import (pca_candidates, add_noise, add_outliers, apply_T, crop_halfspace, estimate_normals, icp,
                    invert_T, make_scene, make_T, pose_error, register, so3_exp)

AXIS = np.array([0.48, -0.6, 0.64])


def gt(angle_deg, t=(0.2, -0.1, 0.15), axis=AXIS):
    return make_T(so3_exp(np.radians(angle_deg) * np.asarray(axis)), np.asarray(t))


class TestICP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.dst, cls.dn = make_scene(2000, rng)
        cls.src_world, _ = make_scene(1600, rng)   # independent sample of the same surface
        cls.T = gt(20)
        cls.src = apply_T(invert_T(cls.T), cls.src_world)

    def test_exact_correspondence_recovers_pose_to_machine_precision(self):
        # src is literally a rigid copy of dst: the only fixed point is the truth
        src = apply_T(invert_T(self.T), self.dst)
        r = icp(src, self.dst, method="point_to_point", max_iter=200)
        rot, tr = pose_error(r.T, self.T)
        self.assertLess(rot, 1e-6)
        self.assertLess(tr, 1e-8)
        self.assertLess(r.rmse, 1e-8)

    def test_point_to_plane_recovers_pose_from_independent_samples(self):
        r = icp(self.src, self.dst, method="point_to_plane", dst_normals=self.dn)
        rot, tr = pose_error(r.T, self.T)
        self.assertTrue(r.converged)
        self.assertLess(rot, 0.1)
        self.assertLess(tr, 2e-3)

    def test_point_to_plane_needs_far_fewer_iterations(self):
        p2p = icp(self.src, self.dst, method="point_to_point", max_iter=300)
        p2l = icp(self.src, self.dst, method="point_to_plane", dst_normals=self.dn)
        self.assertLess(pose_error(p2p.T, self.T)[0], 2.0)  # converges, but slowly and biased
        self.assertLess(3 * p2l.iterations, p2p.iterations)
        self.assertLess(pose_error(p2l.T, self.T)[0], pose_error(p2p.T, self.T)[0])

    def test_point_to_point_error_is_monotone(self):
        # Besl & McKay 1992, Theorem 1: with full correspondence sets the
        # mean squared pair distance never increases.
        r = icp(self.src, self.dst, method="point_to_point", max_iter=80)
        h = np.array(r.mse_history)
        self.assertTrue(np.all(np.diff(h) <= 1e-12 * h[:-1]))

    def test_noise_with_estimated_normals(self):
        rng = np.random.default_rng(12)
        src = add_noise(self.src, 0.01, rng)
        dst = add_noise(self.dst, 0.01, rng)
        r = icp(src, dst, method="point_to_plane", dst_normals=estimate_normals(dst, k=20))
        rot, tr = pose_error(r.T, self.T)
        self.assertLess(rot, 0.5)
        self.assertLess(tr, 5e-3)

    def test_trimming_rescues_partial_overlap_with_outliers(self):
        rng = np.random.default_rng(13)
        dst_c, dn_c = crop_halfspace(self.dst, self.dn, (1, 0, 0), -0.45)
        keep = self.src_world[:, 0] < 0.95
        src_c = add_outliers(self.src[keep], 0.15, rng)
        plain = icp(src_c, dst_c, method="point_to_plane", dst_normals=dn_c)
        trimmed = icp(src_c, dst_c, method="point_to_plane", dst_normals=dn_c, trim=0.55)
        self.assertLess(pose_error(trimmed.T, self.T)[0], 1.0)
        self.assertLess(pose_error(trimmed.T, self.T)[1], 5e-3)
        self.assertGreater(pose_error(plain.T, self.T)[0], 5 * pose_error(trimmed.T, self.T)[0])

    def test_pca_multistart_handles_large_rotation(self):
        T = gt(150, axis=(0.0, 0.6, 0.8))
        src = apply_T(invert_T(T), self.src_world)
        r = register(src, self.dst, dst_normals=self.dn)
        rot, tr = pose_error(r.T, T)
        self.assertLess(rot, 0.1)
        self.assertLess(tr, 2e-3)

    def test_pca_candidates_are_proper_rotations(self):
        # eigh returns eigenvector frames of either handedness; every seed must
        # still be a rotation, never a reflection
        rng = np.random.default_rng(15)
        for _ in range(10):
            src = rng.normal(size=(200, 3)) * [3.0, 2.0, 1.0]
            dst = src @ so3_exp(rng.normal(size=3)).T
            for T in pca_candidates(src, dst):
                self.assertAlmostEqual(np.linalg.det(T[:3, :3]), 1.0, places=10)

    def test_planar_overlap_is_regularised_not_singular(self):
        # a single plane leaves 3 DoF unconstrained for point-to-plane; the
        # step must stay finite and must not move the points off the plane
        rng = np.random.default_rng(14)
        P = np.c_[rng.uniform(-1, 1, (400, 2)), np.zeros(400)]
        N = np.tile([0, 0, 1.0], (400, 1))
        r = icp(P + [0, 0, 0.1], P, method="point_to_plane", dst_normals=N, max_iter=10)
        self.assertTrue(np.all(np.isfinite(r.T)))
        self.assertAlmostEqual(r.T[2, 3], -0.1, places=6)

    def test_argument_validation(self):
        with self.assertRaises(ValueError):
            icp(self.src, self.dst, method="point_to_plane")
        with self.assertRaises(ValueError):
            icp(self.src, self.dst, method="nope")
        with self.assertRaises(ValueError):
            icp(self.src, self.dst, trim=0)


if __name__ == "__main__":
    unittest.main()
