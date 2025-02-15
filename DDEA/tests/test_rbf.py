#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from src.rbf import RBF, RBF_EnsembleUN, RBF_Ensemble_predictor, RBF_predictor


class TestRBF(unittest.TestCase):
    def test_RBF(self):
        X = np.random.rand(10, 2)
        y = np.sum(X, axis=1)
        Nc = 3
        W2, B2, Centers, Spreads = RBF(X, y, Nc)
        self.assertEqual(W2.shape, (Nc,))
        self.assertTrue(isinstance(B2, float) or isinstance(B2, np.float64))
        self.assertEqual(Centers.shape, (Nc, 2))
        self.assertEqual(Spreads.shape, (Nc,))

    def test_RBF_EnsembleUN(self):
        L = np.hstack((np.random.rand(20, 2), np.random.rand(20, 1)))
        c = 2
        nc = 3
        T = 10
        W, B, C_arr, S_arr = RBF_EnsembleUN(L, c, nc, T)
        self.assertEqual(W.shape, (T, nc))
        self.assertEqual(B.shape, (T,))
        self.assertEqual(C_arr.shape, (T, nc, c))
        self.assertEqual(S_arr.shape, (T, nc))

    def test_RBF_Ensemble_predictor(self):
        L = np.hstack((np.random.rand(20, 2), np.random.rand(20, 1)))
        c = 2
        nc = 3
        T = 10
        W, B, C_arr, S_arr = RBF_EnsembleUN(L, c, nc, T)
        U = np.random.rand(5, c)
        Y = RBF_Ensemble_predictor(W, B, C_arr, S_arr, U, c)
        self.assertEqual(Y.shape, (5, T))

    def test_RBF_predictor(self):
        X = np.random.rand(10, 2)
        y = np.sum(X, axis=1)
        Nc = 3
        W2, B2, Centers, Spreads = RBF(X, y, Nc)
        U = np.random.rand(5, 2)
        Y = RBF_predictor(W2, B2, Centers, Spreads, U)
        self.assertEqual(Y.shape, (5,))


if __name__ == "__main__":
    unittest.main()
