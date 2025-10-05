#!/usr/bin/env python

"""
評価と改善を行うターゲット方策(Target policy)
実際に行動を行う挙動方策(Behaviour policy)
方策オフ型：ターゲット方策と挙動方策を分けて考える
"""

import numpy as np

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

# ===== Expectation =====
e = np.sum(x * pi)
print("E_pi[x]", e)

# ===== Monte Carlo =====
n = 100
samples = []
for _ in range(n):
    s = np.random.choice(x, p=pi)
    samples.append(s)
print("MC: {:.2f} (var: {:.2f})".format(np.mean(samples), np.var(samples)))

# ===== Importance Sampling =====
# bの確率分布をpiの確率分布に形を近づけてみる
b = np.array([0.2, 0.2, 0.6])  # b = np.array([1/3, 1/3, 1/3])
samples = []
for _ in range(n):
    idx = np.arange(len(b))  # [0, 1, 2]
    i = np.random.choice(idx, p=b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(rho * s)
print("IS: {:.2f} (var: {:.2f})".format(np.mean(samples), np.var(samples)))
