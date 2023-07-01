#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import differential_evolution
from rbfnet import RBFNet

# 目的関数
def objective_function(x):
    return x**2 + x + 2

# RBFネットワークの初期化
rbf_net = RBFNet(k=2, lr=0.01, epochs=100)

# データの生成
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = objective_function(X)

# RBFネットワークの学習
rbf_net.fit(X, y)

# 大域的最適化
bounds = [(-5, 5)]
result = differential_evolution(lambda x: rbf_net.predict(np.array([x])), bounds)

# 最適解の表示
print(f"Global minimum: x = {result.x[0]:.3f}, y = {result.y[0]:.3f}")

