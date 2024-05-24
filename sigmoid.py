#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 楕円中のパラメータ
radius_x = 10 # x軸方向の半径
radius_y = 3 # y軸方向の半径
height = 10  # 高さ
num_points = 100

# データ生成
theta = np.linspace(0, 2 * np.pi, num_points)
z = np.linspace(-height / 2, height / 2, num_points)
theta, z = np.meshgrid(theta, z)
x = radius_x * np.cos(theta)
y = radius_y * np.sin(theta)

# シグモイド関数の定義
def sigmoid_3d_ellipse(x, y, z, a, ax, ay, c, d, e, f):
    return 1 / (1 + np.exp(-a * ((x / ax)**2 + (y / ay)**2 - 1) + c*z + d)) * e + f

# パラメータの設定
a = 10 # 変化の急さ
ax = radius_x # x軸方向の半径
ay = radius_y # y軸方向の半径
c = 0
d = 0
e = 1
f = 0

# 近似データの計算
data_approx = sigmoid_3d_ellipse(x, y, z, a, ax, ay, c, d, e, f)

# 3Dメッシュプロット
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(data_approx),
                       rstride=1, cstride=1, alpha=0.8)
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_title("Sigmoid Function Approximation of Elliptical Cylinder")
# 軸のスケールを等しくする
ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
plt.show()
