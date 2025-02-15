#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from src.utils import lhs_design, radbas, cdist


class TestUtils(unittest.TestCase):
    def test_lhs_design(self):
        n = 100
        d = 5
        samples = lhs_design(n, d)
        self.assertEqual(samples.shape, (n, d))
        self.assertTrue(np.all(samples >= 0) and np.all(samples <= 1))

    def test_radbas(self):
        x = np.array([0, 1, 2])
        result = radbas(x)
        expected = np.exp(-(x**2))
        np.testing.assert_almost_equal(result, expected)

    def test_cdist(self):
        A = np.array([[0, 0], [1, 1]])
        B = np.array([[0, 1], [1, 0]])
        distances = cdist(A, B)
        expected = np.array([[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_almost_equal(distances, expected, decimal=5)


if __name__ == "__main__":
    unittest.main()
