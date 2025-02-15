#!/usr/bin/env python

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
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
    c = 5  # 次元数
    n_samples = 50
    bd = np.full(c, -5)
    bu = np.full(c, 5)
    L = generate_offline_data(c, n_samples, bd, bu)

    # DDEA_SEアルゴリズムの実行
    exec_time, best_solution, gbest = DDEA_SE(c, L, bu, bd)

    print("=== ベンチマーク: Sphere関数 ===")
    print("実行時間: {:.4f}秒".format(exec_time))
    print("最終最良解:", best_solution)
    print("最良解の目的関数値:", sphere_function(best_solution.reshape(1, -1))[0])
