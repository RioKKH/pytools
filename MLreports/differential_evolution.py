#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import differential_evolution

# 目的関数の定義 (最小化したい関数)
def func(x):
    return x[0]**2 + x[1]**2

# 変数の範囲を定義 (各変数の下限と上限のペアのリスト)
bounds = [(-5, 5), (-5, 5)]

# 差分進化による最適化
result = differential_evolution(func, bounds)

# 目的関数空間の描画
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = func([X, Y])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, 
                rstride=1, cstride=1,
                alpha=0.3, color='blue', edgecolor='black')

# 最適解の描画
ax.scatter(result.x[0], result.x[1], result.fun, color='red', s=100)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()

# 結果の表示
print("最適解: ", result.x)
print("目的関数の値: ", result.fun)
