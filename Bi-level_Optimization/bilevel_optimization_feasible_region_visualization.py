#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def company_profit(x, y):
    return x * y - 100

def consumer_utility(y, x):
    return 10 * np.log(y + 1) - x * y

def optimal_demand(x):
    res = minimize_scalar(lambda y: -consumer_utility(y, x),
                          bounds=(0, 100),
                          method="bounded")
    return res.x

# データ点の生成
x = np.linspace(0.1, 10, 100)
y_optimal = np.array([optimal_demand(xi) for xi in x])

# 上位問題の目的関数値(利益)を計算
z_profit = company_profit(x, y_optimal)

# グラフの描画
plt.figure(figsize=(12, 8))

# 実行可能領域
plt.fill_between(x, y_optimal, 0, alpha=0.3, color="gray", label="Feasible Region")


# 等高線 (上位問題の目的関数)
Y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, Y)
Z = company_profit(X, Y)
contour = plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
plt.colorbar(contour, label="Company Profit")

# 実行可能な最適解の軌跡 
plt.plot(x, y_optimal, "r-", label="Optimal Demand (Lower Level Solution)")

# 全体の最適解 (制約なし)
y_global, x_global = np.unravel_index(np.argmax(Z), Z.shape)
#x_global, y_global = np.unravel_index(np.argmax(Z), Z.shape)
#plt.plot(x[x_global], Y[y_global], "go", markersize=10, label="Global Optimum (Unconstrained)")
plt.plot(x[x_global], Y[y_global, x_global], "go", markersize=10, label="Global Optimum (Unconstrained)")

# Bi-level 最適解
bi_level_optimal = np.argmax(z_profit)
plt.plot(x[bi_level_optimal], y_optimal[bi_level_optimal], "bo",
         markersize=10, label="Bi-level Optimum")

plt.xlabel("Price (x)")
plt.ylabel("Demand (y)")
plt.title("Bi-level Optimization: Feasible Region and Optima")
plt.legend()
plt.grid(True)
plt.show()

print(f"Bi-level最適解: 価格 = {x[bi_level_optimal]:.2f}, 需要 = {y_optimal[bi_level_optimal]:.2f}")
print(f"Bi-level最適利益: {z_profit[bi_level_optimal]:.2f}")
print(f"\n全体最適解 (制約なし) : 価格 = {x[x_global]:.2f}, 需要 = {Y[y_global, 0]:.2f}")
print(f"全体最適利益 (制約なし): {Z[x_global, y_global]:.2f}")
