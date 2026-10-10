import unittest

import numpy as np

from icpkit import estimate_normals, knn, nearest_neighbors, sample_box, sample_sphere, sample_torus


class TestSamplers(unittest.TestCase):
    def test_torus_points_satisfy_implicit_equation_and_are_area_uniform(self):
        rng = np.random.default_rng(0)
        P, N = sample_torus(40000, rng, R=1.0, r=0.3)
        rho = np.hypot(P[:, 0], P[:, 1])
        np.testing.assert_allclose((rho - 1.0) ** 2 + P[:, 2] ** 2, 0.09, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-12)
        # area element is r (R + r cos v) du dv.  The outer half (cos v > 0) has
        # integral over v of pi R + 2 r against 2 pi R in total, so its share
        # of the area is 0.5 + r / (pi R) = 0.5955 for R=1, r=0.3.
        frac_outer = np.mean(rho > 1.0)
        self.assertAlmostEqual(frac_outer, 0.5 + 0.3 / np.pi, delta=0.01)

    def test_box_points_on_faces_proportional_to_area(self):
        rng = np.random.default_rng(1)
        P, N = sample_box(60000, rng, half=(1.0, 0.5, 0.25))
        on_x = np.isclose(np.abs(P[:, 0]), 1.0)
        # the pair of faces normal to axis i has area proportional to the
        # product of the other two half-extents
        areas = np.array([0.5 * 0.25, 1.0 * 0.25, 1.0 * 0.5])
        self.assertAlmostEqual(on_x.mean(), areas[0] / areas.sum(), delta=0.01)
        np.testing.assert_allclose(np.abs(N).sum(1), 1.0)


class TestNeighbours(unittest.TestCase):
    def test_nearest_neighbors_matches_loop(self):
        rng = np.random.default_rng(2)
        A, B = rng.normal(size=(300, 3)), rng.normal(size=(500, 3))
        idx, dist = nearest_neighbors(A, B, chunk=64)
        for i in range(300):
            d = np.linalg.norm(B - A[i], axis=1)
            self.assertEqual(idx[i], np.argmin(d))
            self.assertAlmostEqual(dist[i], d.min(), places=12)

    def test_knn_first_neighbour_is_self_and_sorted(self):
        rng = np.random.default_rng(3)
        P = rng.normal(size=(400, 3))
        nb = knn(P, 8)
        np.testing.assert_array_equal(nb[:, 0], np.arange(400))
        d = np.linalg.norm(P[nb] - P[:, None, :], axis=2)
        self.assertTrue(np.all(np.diff(d, axis=1) >= -1e-12))


class TestNormals(unittest.TestCase):
    def test_plane_normals(self):
        rng = np.random.default_rng(4)
        P = np.c_[rng.uniform(-1, 1, (500, 2)), np.zeros(500)]
        N = estimate_normals(P, k=12, viewpoint=(0, 0, 5))
        np.testing.assert_allclose(N, np.tile([0, 0, 1.0], (500, 1)), atol=1e-9)

    def test_sphere_normals_are_radial(self):
        rng = np.random.default_rng(5)
        P, N_true = sample_sphere(3000, rng)
        N = estimate_normals(P, k=16)
        cos = (N * N_true).sum(1)
        self.assertGreater(cos.min(), 0.99)


if __name__ == "__main__":
    unittest.main()
