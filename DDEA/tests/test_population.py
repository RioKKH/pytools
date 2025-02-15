#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from src.population import initialize_pop


class TestPopulation(unittest.TestCase):
    def test_initialize_pop(self):
        n = 50
        c = 4
        bd = np.zeros(c)
        bu = np.ones(c) * 10
        POP = initialize_pop(n, c, bu, bd)
        self.assertEqual(POP.shape, (n, c))
        self.assertTrue(np.all(POP >= bd))
        self.assertTrue(np.all(POP <= bu))


if __name__ == "__main__":
    unittest.main()
