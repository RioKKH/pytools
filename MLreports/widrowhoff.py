#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

X = np.array([1.0, 0.5, -0.2, -0.4, -1.3, -2.0])
y = np.array([1, 1, 0, 1, 0, 0])

n, d = X.shape[0], 1
print(n, d) # 6 1

X = np.vstack((np.ones(n), X)).T
print(X)
 #[ x0   x1  ]
 #[[ 1.   1. ]
 #[  1.   0.5]
 #[  1.  -0.2]
 #[  1.  -0.4]
 #[  1.  -1.3]
 #[  1.  -2. ]]   

eps = 1e-8
differ = np.inf
olderr = np.inf
w = np.array([0.2, 0.3])
rho = 0.2
m = 1

while differ > eps:
    w = w - rho * np.sum(X * np.repeat((X @ w - y)[:, np.newaxis], 2, axis=1), axis=0)
    # 1. X @ w: Xはデータの行列で、wはパラメータのベクトル。X @ wは各データに対する予測値
    # 2. (X @ w - y)[:, np.newaxis] 誤差を求める。子のベクトルを列べくトロに変換する
    #    各行に1つの要素をもつ2次元配列にする
    # 3. np.repeat(..., 2, axis=1): 列ベクトルを横に2回繰り返して2列を持つ行列を作る
    #    これにより誤差が獲得超量に対して繰り返される
    # 4. X * ...: 各データの各特徴量に対応する誤差を掛けている。
    # 5. np.sum(..., axis=0): 獲得超量に対する誤差を全てのデータについて合計している。
    #    axis=0で指定しているのは、行、つまりデータについての合計を計算するため。
    # 6. w - rho * ...: 学習率rhoを掛けた誤差を現在のパラメータから引いて、パラメータを更新
    # 7. 予測が真の値よりも大きければパラメータを小さくし、予測が真の値よりも
    #    小さければパラメータを大きくする。
    sqrerr = 0.5 * np.sum((X @ w - y) ** 2)
    differ = abs(olderr - sqrerr)
    olderr = sqrerr
    print(f"step = {m}, w0 = {w[0]:6.3f}, w1 = {w[1]:6.3f}, err = {sqrerr:11.8f}")
    m += 1

print(f"Results: w0 = {w[0]:6.3f}, w1 = {w[1]:6.3f}")
