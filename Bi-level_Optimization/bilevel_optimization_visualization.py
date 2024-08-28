#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def company_profit(x, y):
    return x * y - 100

def consumer_utility(y, x):
    return 10 * np.log(y + 1) - x * y

# データ点の生成
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)

# 上位問題 (企業の利益) のグラフ
Z1 = company_profit(X, Y)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis')
ax1.set_xlabel('Price (x)')
ax1.set_ylabel('Demand (y)')
ax1.set_zlabel('Profit')
ax1.set_title('Upper Level: Company Profit')
fig.colorbar(surf1, shrink=0.5, aspect=5)


# 下位問題 (消費者の効用) のグラフ
Z2 = consumer_utility(Y, X)

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z2, cmap='plasma')
ax2.set_xlabel('Price (x)')
ax2.set_ylabel('Demand (y)')
ax2.set_zlabel('Utility')
ax2.set_title('Lower Level: Consumer Utility')
fig.colorbar(surf2, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 最適解の探索 (簡易的なグリッドサーチ)
best_profit = float('-inf')
best_x = best_y = 0

for price in np.linspace(0, 10, 50):
    for demand  in np.linspace(0, 10, 50):
        utility = consumer_utility(demand, price)
        if utility > consumer_utility(best_y, price):
            profit = company_profit(price, demand)
            if profit > best_profit:
                best_profit = profit
                best_x, best_y = price, demand

print(f"最適価格: {best_x:.2f}")
print(f"最適需要: {best_y:.2f}")
print(f"最大利益: {best_profit:.2f}")
