#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from src.operators import SBX, mutation


class TestOperators(unittest.TestCase):
    def test_SBX(self):
        pop = np.random.rand(10, 3)
        bu = np.ones(3) * 5
        bd = np.zeros(3)
        NPOP = SBX(pop, bu, bd, 1.0, 5)
        self.assertEqual(NPOP.shape, (10, 3))
        self.assertTrue(np.all(NPOP >= bd))
        self.assertTrue(np.all(NPOP <= bu))

    def test_mutation(self):
        pop = np.random.rand(10, 3)
        bu = np.ones(3) * 5
        bd = np.zeros(3)
        NPOP = mutation(pop, bu, bd, 0.5, 10)
        self.assertEqual(NPOP.shape, (10, 3))
        self.assertTrue(np.all(NPOP >= bd))
        self.assertTrue(np.all(NPOP <= bu))


if __name__ == "__main__":
    unittest.main()
