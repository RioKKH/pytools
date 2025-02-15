#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from src.ddea_se import DDEA_SE


class TestDDEA_SE(unittest.TestCase):
    def test_run(self):
        c = 3
        n_samples = 20
        bd = np.zeros(c)
        bu = np.ones(c) * 5
        # オフラインデータ（Sphere関数 f(x)=sum(x^2) を例として）
        X = np.random.uniform(bd, bu, size=(n_samples, c))
        y = np.sum(X**2, axis=1)
        L = np.hstack((X, y.reshape(-1, 1)))
        exec_time, P, gbest = DDEA_SE(c, L, bu, bd)
        self.assertTrue(exec_time > 0)
        self.assertEqual(P.shape, (c,))
        self.assertEqual(gbest.shape[1], c)

    def test_run_2d_history(self):
        # ２次元のテスト (record_history=Trueの場合)
        c = 2
        n_samples = 30
        bd = np.full(c, -5)
        bu = np.full(c, 5)
        X = np.random.uniform(bd, bu, size=(n_samples, c))
        y = np.sum(X**2, axis=1)
        L = np.hstack((X, y.reshape(-1, 1)))
        # record_history=Trueの場合、返り値は５要素となるはず
        result = DDEA_SE(c, L, bu, bd, record_history=True)
        self.assertEqual(len(result), 5)
        exec_time, P, gbest, pop_history, surrogate_history = result
        self.assertTrue(exec_time > 0)
        self.assertEqual(P.shape, (c,))
        self.assertEqual(gbest.shape[1], c)


if __name__ == "__main__":
    unittest.main()
