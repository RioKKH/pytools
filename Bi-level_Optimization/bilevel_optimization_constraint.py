#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 上位問題: f(x, y) = (x - 3)^2 + (y - 2)^2を最小化
def upper_problem(x, y):
    return (x - 3)**2 + (y - 2)**2

# 下位問題: g(y, x) = (y - x)^2を最小化
def lower_problem(y, x):
    return (y - x)**2

# 下位問題の最適解
def lower_problem_optimal(x):
    return x # y = xが最適解

# データ点の生成
x = np.linspace(0, 6, 100)
y = np.linspace(0, 6, 100)
X, Y = np.meshgrid(x, y)

# 上位問題の等高線
Z_upper = upper_problem(X, Y)

# グラフの描画    
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z_upper, levels=20, cmap='viridis')
plt.colorbar(label='Upper problem objective')

# 下位問題の最適解 (制約)
plt.plot(x, lower_problem_optimal(x), 'r--', 
         label='Lower problem optimal solution (constraint)')

# 上位問題のしんの最適解 (制約を無視した場合)
plt.plot(3, 2, 'ro', label='True optimum (ignoring constraint)')

# Bi-level 最適解の探索
best_x, best_y = min(((xi, lower_problem_optimal(xi)) for xi in x),
                     key=lambda p: upper_problem(p[0], p[1]))
plt.plot(best_x, best_y, 'go', label='Bi-level optimum')

plt.plot(best_x, best_y, 'go', label='Bi-level optimum')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Bi-level Optimization: Constraint Visualization')
plt.legend()
plt.grid(True)
plt.show()


print(f"Bi-level最適解: x={best_x:.2f}, y={best_y:.2f}")
print(f"上位問題の目的関数値: {upper_problem(best_x, best_y):.2f}")
print(f"下位問題の目的関数値: {lower_problem(best_y, best_x):.2f}")

