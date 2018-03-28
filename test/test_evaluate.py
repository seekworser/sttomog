import unittest
import numpy as np
import sttomog


class TestEvaluate(unittest.TestCase):
    def test_fidelity(self):
        rho = np.array([
            [0, 0, 0, 0],
            [0, .5, -.5, 0],
            [0, -.5, .5, 0],
            [0, 0, 0, 0],
        ])
        sigma = np.eye(4) / 4.
        self.assertRaises(ValueError)
        self.assertAlmostEqual(sttomog.fidelity(rho, sigma), .5, places=7)

    def test_concurrence(self):
        rho = np.array([
            [0, 0, 0, 0],
            [0, .5, -.5, 0],
            [0, -.5, .5, 0],
            [0, 0, 0, 0],
        ])
        sigma = np.eye(4) / 4.
        self.assertRaises(ValueError)
        self.assertAlmostEqual(sttomog.concurrence(rho), 1.0, places=7)
        self.assertAlmostEqual(sttomog.concurrence(sigma), 0.0, places=7)
