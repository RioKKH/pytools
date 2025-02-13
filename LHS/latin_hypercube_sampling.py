#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def latin_hypercube_sampling(n_samples, n_dim):
    """
    Latin Hypercube Samplingを実施する関数

    Parameters:
        n_samples (int): サンプル数
        n_dim (int): 次元数

    Returns:
        samples (numpy.ndarray): (n_samples x n_dim)のサンプル行列

    Explanations:
        各次元の値域を同じ大きさの区間に分割し、各区間から1点ずつ
        サンプリングする。サンプル数をNとすると、各次元の値域[0, 1]
        を以下の様にN個の区間に分割する
        [0, 1/N), [1/N, 2/N), cdots, [(N-1)/N, 1)
        次に、各次元について以下の手順を実施する
        1. ランダムな順列を生成
            各次元jに対して、1からNまでの整数のランダムな順列pi_jを
            生成する。これにより、各区間が1度だけ利用されるように
            割り当てが行われる
        2. 各区間内でのランダムサンプリング
            各サンプルiに対して、次元jの値は、ランダムなオフセットを
            持ってその区間内から選ばれる。すなわち、サンプルiの次元j
            に対する値x_{ij}は、以下の様に計算される
            x_{ij} = frac{pi_{j}(i) - qsi_{ij}}{N} 但し、qsi_{ij} ~ uniform(0, 1)
            ここで、pi_j(i)は次元jにおけるi番目の順列の値
            qsi_{ij}は区間内のランダムなオフセット(0~1の一様乱数)
    """

    samples = np.zeros((n_samples, n_dim))
    # 各次元毎に処理を行う
    for j in range(n_dim):
        # 1からn_samplesまでのランダムな順列を生成
        perm = np.random.permutation(n_samples)
        # 各区間内でランダムな値を加える
        samples[:, j] = (perm + np.random.uniform(size=n_samples)) / n_samples

    return samples


# サンプル数と次元数の設定
n_samples = 10  # 例として50点を生成
n_dim = 2  # 2次元の場合

# Latin Hypercube Samplingの実行
samples = latin_hypercube_sampling(n_samples, n_dim)
print("生成されたサンプル:")
print(samples)

# 生成されたサンプルをプロット
plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], c="blue", edgecolors="k")
plt.title("Latin Hypercube Sampling (2-Dim)")
plt.xlabel("Dim 1")
plt.xlabel("Dim 2")
plt.grid(True)
plt.show()
