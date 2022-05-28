#!/usr/bin/env python
import unittest

import numpy as np
from scipy.linalg import expm

import liegroup


class TestSO2(unittest.TestCase):
    def test_default(self):
        R_12 = liegroup.SO2()
        self.assertTrue((R_12.matrix() == np.eye(2)).all())

    def test_exp(self):
        theta = 0.25
        R_12_se2 = liegroup.SO2.exp(theta).matrix()
        R_12_exp = expm(liegroup.SO2.hat(theta))
        self.assertAlmostEqual(np.abs(R_12_se2 - R_12_exp).max(), 0, places=9)

    def test_exp_identity(self):
        R_12 = liegroup.SO2.exp(0)
        self.assertTrue((R_12.matrix() == np.eye(2)).all())

    def test_log(self):
        theta = np.pi / 4
        self.assertAlmostEqual(liegroup.SO2.exp(theta).log(), theta, places=9)

    def test_adj(self):
        R_12 = liegroup.SO2.exp(0)
        self.assertEqual(R_12.adj(), 1.0)

    def test_inverse(self):
        R_12 = liegroup.SO2.exp(np.pi / 6)

        res = R_12 @ R_12.inverse()
        self.assertTrue((res.matrix() == np.eye(2)).all())

        res = R_12.inverse() @ R_12
        self.assertTrue((res.matrix() == np.eye(2)).all())

    def test_matmul_point(self):
        # With angle pi/2, x2 = -y1, y2 = x1.
        R_12 = liegroup.SO2.exp(np.pi / 2)
        res = R_12 @ [1, 3]
        self.assertAlmostEqual(res[0], -3, places=9)
        self.assertAlmostEqual(res[1], 1, places=9)

    # XXX test matmul point as list
    # XXX test matmul array of points


class TestSE2(unittest.TestCase):
    def test_default(self):
        transform_12 = liegroup.SE2()
        self.assertTrue((transform_12.so2.matrix() == np.eye(2)).all())
        self.assertTrue((transform_12.p == np.zeros(2)).all())

    def test_exp(self):
        theta = 0.25
        p = np.array([-5, 3])
        phi = np.hstack([p, theta])
        transform_12_se2 = liegroup.SE2.exp(phi)
        G = expm(liegroup.SE2.hat(phi))
        self.assertAlmostEqual(np.abs(transform_12_se2.so2.matrix() - G[:2, :2]).max(), 0, places=9)
        self.assertAlmostEqual(np.abs(transform_12_se2.p - G[:2, 2]).max(), 0, places=9)

    def test_exp_identity(self):
        transform_12 = liegroup.SE2.exp(np.zeros(3))
        self.assertTrue((transform_12.so2.matrix() == np.eye(2)).all())
        self.assertTrue((transform_12.p == np.zeros(2)).all())

    def test_log(self):
        phi = np.hstack([[-5, 3], np.pi / 6])
        self.assertAlmostEqual(np.abs(liegroup.SE2.exp(phi).log() - phi).max(), 0, places=9)

    def test_adj(self):
        transform_12 = liegroup.SE2(so2=liegroup.SO2.exp(np.pi / 4),
                                    p=np.array([1, 2]))
        # exp(Ad_X eps) = X exp(eps) inv(X)
        eps = np.hstack([[-3, 5], -np.pi/6])
        res_lhs = liegroup.SE2.exp(transform_12.adj() @ eps)
        res_rhs = transform_12 @ liegroup.SE2.exp(eps) @ transform_12.inverse()
        self.assertAlmostEqual(np.abs(res_lhs.so2.matrix() - res_rhs.so2.matrix()).max(),
                               0.0, places=9)
        self.assertAlmostEqual(np.abs(res_lhs.p - res_rhs.p).max(),
                               0.0, places=9)

    def test_inverse(self):
        phi = np.hstack([[-5, 3], np.pi / 6])
        transform_12 = liegroup.SE2.exp(phi)

        res = transform_12 @ transform_12.inverse()
        self.assertTrue((res.so2.matrix() == np.eye(2)).all())
        self.assertTrue((res.p == np.zeros(2)).all())

        res = transform_12.inverse() @ transform_12
        self.assertTrue((res.so2.matrix() == np.eye(2)).all())
        self.assertTrue((res.p == np.zeros(2)).all())

    def test_matmul_point(self):
        p_12_1 = np.array([3, 4])

        # Frame 2 wrt frame 1.
        transform_12 = liegroup.SE2(so2=liegroup.SO2.exp(np.pi / 4),
                                    p=p_12_1)

        # A point coincident with the frame 2 origin expressed in frame 2 should
        # be the same as the frame 2 origin expressed in frame 1.
        res = transform_12 @ [0, 0]
        self.assertAlmostEqual(res[0], p_12_1[0], places=9)
        self.assertAlmostEqual(res[1], p_12_1[1], places=9)

        # A point coincident with the frame 2 origin expressed in frame 1 should
        # be the same as the frame 1 origin.
        res = transform_12.inverse() @ p_12_1
        self.assertAlmostEqual(res[0], 0, places=9)
        self.assertAlmostEqual(res[1], 0, places=9)

        # Express a point not at frame 2 origin in frame 1 frame.
        res = transform_12 @ [1, 0]
        self.assertAlmostEqual(res[0], p_12_1[0] + np.sqrt(2) / 2, places=9)
        self.assertAlmostEqual(res[1], p_12_1[1] + np.sqrt(2) / 2, places=9)

    # XXX test matmul point as list
    # XXX test matmul array of points


if __name__ == '__main__':
    unittest.main()
