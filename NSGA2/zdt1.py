#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from nsga2.problem import Problem
from nsga2.evolution import Evolution


def f1(x):
    return x[0]

def f2(x):
    num = len(x)
    g = 1 + 9.0 * np.sum(x[1:]) / (num - 1)
    h = 1 - np.sqrt(x[0] / g)
    return g * h


problem = Problem(num_of_variables=30,
                  objectives=[f1, f2],
                  variables_range=[(0, 1)],
                  same_range=True, expand=False)
evo = Evolution(problem)
evol = evo.evolve()
func = [i.objectives for i in evol]

function1 = [i[0] for i in func]
function2 = [i[1] for i in func]

plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()
