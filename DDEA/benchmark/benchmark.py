#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ddea_se import DDEA_SE


def sphere_function(x):
    """
    Sphere関数: f(x) = sum(x^2)
    """
    return np.sum(x**2, axis=1)


def generate_offline_data(c, n_samples, bd, bu):
    """
    ベンチマーク用に、ランダムサンプルと
    Sphere関数の値からなるオフラインデータを生成
    """
    X = np.random.uniform(bd, bu, size=(n_samples, c))
    y = sphere_function(X)
    L = np.hstack((X, y.reshape(-1, 1)))
    return L


if __name__ == "__main__":
    # ベンチマーク問題のパラメータ設定
    # 2次元問題の設定
    c = 2  # 次元数
    n_samples = 50
    bd = np.full(c, -5)
    bu = np.full(c, 5)
    L = generate_offline_data(c, n_samples, bd, bu)

    # historyを記録してDDEA_SEアルゴリズムの実行
    print("before ddea-se")
    exec_time, best_solution, gbest, pop_history, surrogate_history = DDEA_SE(
        c, L, bu, bd, record_history=True
    )
    print("after ddea-se")

    print("=== ベンチマーク: Sphere関数 (2次元) ===")
    print("実行時間: {:.4f}秒".format(exec_time))
    print("最終最良解:", best_solution)
    print("最良解の目的関数値:", sphere_function(best_solution.reshape(1, -1))[0])

    # 2次元の真の目的関数の等高線 (Sphere関数) を予め計算
    x1 = np.linspace(bd[0], bu[0], 200)
    x2 = np.linspace(bd[1], bu[1], 200)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.c_[X1.ravel(), X2.ravel()]
    Z_true = sphere_function(grid_points).reshape(X1.shape)

    num_generations = len(pop_history)

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        ax.clear()
        # 真の目的関数の等高線
        cs_true = ax.contour(X1, X2, Z_true, levels=30, cmap="gray", alpha=0.5)
        ax.clabel(cs_true, inline=1, fontsize=8)

        # サロゲートモデルの等高線 (その世代の記録)
        X1_s, X2_s, Z_surrogate = surrogate_history[frame]
        cs_sur = ax.contour(
            X1_s, X2_s, Z_surrogate, levels=30, cmap="viridis", alpha=0.7
        )
        ax.clabel(cs_sur, inline=1, fontsize=8)

        # 現在の個体群の散布図
        pop = pop_history[frame]
        ax.scatter(pop[:, 0], pop[:, 1], color="red", marker="o", label="Population")

        # 現在の世代と最良解の表示
        best = gbest[frame]
        ax.scatter(best[0], best[1], color="blue", marker="*", s=150, label="Best")
        ax.set_title(f"Generation {frame + 1}/{num_generations}")
        ax.set_xlim(bd[0], bu[0])
        ax.set_ylim(bd[1], bu[1])
        ax.legend()

    ani = animation.FuncAnimation(
        fig, update, frames=num_generations, interval=200, repeat_delay=1000
    )

    # アニメーションを保存 (mp4またはgif)
    output_file = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "benchmark",
        # "ddea_animation.gif",
        "ddea_animation.mp4",
    )
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=5, metadata=dict(artist="RioKKH"), bitrate=1800)
    ani.save(output_file, writer=writer)

    plt.show()

    # # 各世代の最良解の目的関数値を計算
    # convergence = [sphere_function(g.reshape(1, -1))[0] for g in gbest]

    # plt.figure(figsize=(8, 5))
    # plt.plot(convergence, marker="o")
    # plt.xlabel("Generation")
    # plt.ylabel("Objective Value")
    # plt.title("Convergence Curve on Sphere Function")
    # plt.grid(True)
    # plt.show()
