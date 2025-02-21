#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

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
    # コマンドライン引数による出力モードの選択
    parser = argparse.ArgumentParser(
        description="Output mode: animation or per-generation PNG images"
    )
    parser.add_argument(
        "--mode",
        choices=["animation", "png"],
        default="animation",
        help="出力モード: アニメーション保存か、各世代ごとのPNG画像保存",
    )
    args = parser.parse_args()
    mode = args.mode

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

    # fig, ax = plt.subplots(figsize=(8, 6))
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection="3d")  # 一番左: 実際の目的関数の3Dプロット
    ax2 = fig.add_subplot(
        132, projection="3d"
    )  # 中央:   サロゲートモデルの3Dプロット(個体群とBestを含む)
    ax3 = fig.add_subplot(133)  # 一番右: 従来の2Dコンター図

    def update(frame):
        # 各サブプロットの内容を更新する
        ax1.clear()
        ax2.clear()
        ax3.clear()

        # サブプロット1: 実際の目的関数の3Dプロット
        ax1.plot_surface(X1, X2, Z_true, cmap="gray", alpha=0.8)
        ax1.set_title("Actual Objective Function")
        ax1.set_xlim(bd[0], bu[0])
        ax1.set_ylim(bd[1], bu[1])
        ax1.set_zlabel("f(x)")

        # サブプロット2: サロゲートモデルの3Dプロット
        X1_s, X2_s, Z_surrogate = surrogate_history[frame]
        ax2.plot_surface(X1_s, X2_s, Z_surrogate, cmap="viridis", alpha=0.8)
        # 現在の個体群
        pop = pop_history[frame]
        # 各個体のZ座標は補間(線形補間)
        pop_z = griddata(
            (X1_s.ravel(), X2_s.ravel()),
            Z_surrogate.ravel(),
            (pop[:, 0], pop[:, 1]),
            method="linear",
        )
        ax2.scatter(
            pop[:, 0],
            pop[:, 1],
            pop_z,
            color="red",
            marker="o",
            s=50,
            label="Population",
        )
        # サロゲートアンサンブルでのBest個体
        best = gbest[frame]
        best_z = griddata(
            (X1_s.ravel(), X2_s.ravel()),
            Z_surrogate.ravel(),
            ([best[0], best[1]]),
            method="linear",
        )
        ax2.scatter(
            best[0], best[1], best_z, color="blue", marker="*", s=150, label="Best"
        )
        ax2.set_title("Surrogate Model")
        ax2.set_xlim(bd[0], bu[0])
        ax2.set_ylim(bd[1], bu[1])
        ax2.set_zlabel("Surrogate f(x)")

        # サブプロット3: 従来の2Dコンター図
        # 真の目的関数の等高線
        cs_true = ax3.contour(X1, X2, Z_true, levels=30, cmap="gray", alpha=0.5)
        ax3.clabel(cs_true, inline=1, fontsize=8)

        # サロゲートモデルの等高線 (その世代の記録)
        cs_sur = ax3.contour(
            X1_s, X2_s, Z_surrogate, levels=30, cmap="viridis", alpha=0.7
        )
        ax3.clabel(cs_sur, inline=1, fontsize=8)
        ax3.scatter(pop[:, 0], pop[:, 1], color="red", marker="o", label="Population")
        ax3.scatter(best[0], best[1], color="blue", marker="*", s=150, label="Best")
        ax3.set_title(f"Generation {frame + 1}/{num_generations}")
        ax3.set_xlim(bd[0], bu[0])
        ax3.set_ylim(bd[1], bu[1])
        # 同一レンジの場合、XとYの比率をそろえる
        ax3.set_aspect("equal", adjustable="box")
        ax3.legend(loc="upper right")

    # modeに応じた出力処理
    if mode == "animation":
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
    else:
        # 各世代のPNG画像保存用ディレクトリを作成
        output_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "benchmark",
            "png",
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 各世代毎に update() を実行して画像保存
        for frame in range(num_generations):
            update(frame)
            plt.savefig(os.path.join(output_dir, f"generation_{frame + 1:03d}.png"))

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
