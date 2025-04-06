#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


# 非線形な状態遷移関数
def f(x):
    return x + np.sin(0.05 * x)


# 観測関数(非線形)
def h(x):
    return x**2 / 20


# 初期設定
N = 2000  # パーティクルの数
particles = np.random.normal(loc=0.0, scale=5.0, size=N)  # 初期パーティクル
weights = np.ones(N) / N  # 初期重み

true_state = 0
observations = []
estimates = []

# シミュレーションの実行
for t in range(50):
    # 真の状態更新
    true_state = f(true_state) + np.random.normal(0, 1.0)
    # 観測地を得る(ノイズ付き)
    z = h(true_state) + np.random.normal(0, 1.0)
    observations.append(z)

    # 予測ステップ
    particles = f(particles) + np.random.normal(0, 1.0, size=N)

    # 更新ステップ (重み計算)
    weights *= (1 / np.sqrt(2 * np.pi)) * np.exp(-((z - h(particles)) ** 2) / 2)
    weights += 1.0e-300  # 数値的安定性のために小さな値を加える
    weights /= sum(weights)  # 正規化

    # 推定値の算出 (重み付き平均)
    estimate = np.sum(particles * weights)
    estimates.append(estimate)

    # リサンプリング (系統サンプリング)
    indices = np.random.choice(N, N, p=weights)
    particles = particles[indices]
    weights.fill(1.0 / N)

# 結果の描画
plt.figure(figsize=(10, 6))
plt.plot(observations, "ro", label="Observations")
plt.plot(estimates, "b-", linewidth=2, label="Particle Filter Estimate")
plt.legend()
plt.xlabel("Time step")
plt.ylabel("State")
plt.title("Particle Filter Tracking (Nonlinear)")
plt.grid()
plt.show()
