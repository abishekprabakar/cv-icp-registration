"""Closed-form geometry checks with hand-computed or independent expected values."""
import unittest

import numpy as np

from icpkit import (apply_T, horn_quaternion, invert_T, kabsch, make_T, rotation_error_deg,
                    random_rotation, so3_exp, so3_log, umeyama)

RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class TestSO3(unittest.TestCase):
    def test_rodrigues_90_about_z_by_hand(self):
        np.testing.assert_allclose(so3_exp(np.array([0, 0, np.pi / 2])), RZ90, atol=1e-15)

    def test_rodrigues_120_about_diagonal_is_axis_cycle(self):
        # 120 deg about (1,1,1)/sqrt3 permutes the axes x->y->z->x
        R = so3_exp(np.ones(3) / np.sqrt(3) * 2 * np.pi / 3)
        np.testing.assert_allclose(R, [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-14)

    def test_exp_log_roundtrip_including_near_pi(self):
        rng = np.random.default_rng(1)
        for angle in [1e-9, 1e-3, 0.5, 2.0, 3.0, np.pi - 1e-7]:
            a = rng.normal(size=3); a /= np.linalg.norm(a)
            R = so3_exp(a * angle)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(so3_exp(so3_log(R)), R, atol=1e-6 if angle > 3.1 else 1e-12)

    def test_geodesic_angle(self):
        self.assertAlmostEqual(rotation_error_deg(np.eye(3), RZ90), 90.0, places=10)

    def test_invert_T(self):
        T = make_T(RZ90, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(T @ invert_T(T), np.eye(4), atol=1e-15)


class TestAbsoluteOrientation(unittest.TestCase):
    def test_kabsch_hand_example(self):
        # origin and unit axes, rotated 90 deg about z and shifted by (1,2,3)
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        Q = np.array([[1, 2, 3], [1, 3, 3], [0, 2, 3], [1, 2, 4.0]])
        R, t = kabsch(P, Q)
        np.testing.assert_allclose(R, RZ90, atol=1e-14)
        np.testing.assert_allclose(t, [1, 2, 3], atol=1e-14)

    def test_kabsch_never_returns_reflection(self):
        rng = np.random.default_rng(2)
        P = rng.normal(size=(30, 3))
        Q = P * np.array([1, 1, -1])  # mirror image: best orthogonal fit is a reflection
        R, _ = kabsch(P, Q)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_kabsch_matches_horn_quaternion_on_noisy_weighted_data(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            P = rng.normal(size=(50, 3))
            Rt = random_rotation(rng)
            Q = P @ Rt.T + rng.normal(size=3) + 0.05 * rng.normal(size=(50, 3))
            w = rng.uniform(0.1, 2.0, size=50)
            R1, t1 = kabsch(P, Q, w)
            R2, t2 = horn_quaternion(P, Q, w)
            np.testing.assert_allclose(R1, R2, atol=1e-9)
            np.testing.assert_allclose(t1, t2, atol=1e-9)

    def test_kabsch_is_a_stationary_minimum(self):
        rng = np.random.default_rng(4)
        P = rng.normal(size=(40, 3))
        Q = P @ random_rotation(rng).T + 0.1 * rng.normal(size=(40, 3))
        R, t = kabsch(P, Q)
        cost = lambda R_, t_: np.sum((P @ R_.T + t_ - Q) ** 2)
        c0 = cost(R, t)
        for _ in range(50):
            dR = so3_exp(1e-3 * rng.normal(size=3))
            self.assertGreater(cost(dR @ R, t + 1e-3 * rng.normal(size=3)), c0)

    def test_umeyama_recovers_scale(self):
        rng = np.random.default_rng(5)
        P = rng.normal(size=(25, 3))
        Rt = random_rotation(rng)
        s, R, t = umeyama(P, 2.5 * P @ Rt.T + np.array([0.3, -1.0, 4.0]))
        self.assertAlmostEqual(s, 2.5, places=12)
        np.testing.assert_allclose(R, Rt, atol=1e-12)
        np.testing.assert_allclose(t, [0.3, -1.0, 4.0], atol=1e-12)

    def test_apply_T(self):
        T = make_T(RZ90, np.array([1.0, 0, 0]))
        np.testing.assert_allclose(apply_T(T, np.array([[1.0, 0, 0]])), [[1.0, 1.0, 0.0]], atol=1e-15)


if __name__ == "__main__":
    unittest.main()
