#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from src.selection import SelectModels


class TestSelection(unittest.TestCase):
    def test_SelectModels(self):
        T = 20
        Q = 5
        c = 3
        # ダミーのRBFモデルパラメータ（適当な乱数で代用）
        W = np.random.rand(T, 3)
        B = np.random.rand(T)
        C_arr = np.random.rand(T, 3, c)
        S_arr = np.random.rand(T, 3) + 0.1
        Xb = np.random.rand(1, c)
        selected = SelectModels(W, B, C_arr, S_arr, Xb, c, Q)
        self.assertEqual(selected.shape, (Q,))
        self.assertTrue(np.all(selected >= 0) and np.all(selected < T))


if __name__ == "__main__":
    unittest.main()
