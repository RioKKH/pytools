#!/usr/bin/env python

import numpy as np
from numpy.linalg import pinv
from .utils import radbas, cdist


def RBF(SamIn, SamOut, Nc):
    """
    単一RBFモデルの構築

    Parameters:
      SamIn : numpy.ndarray
          学習入力 (形状: (num_samples, d))
      SamOut : numpy.ndarray
          学習出力 (形状: (num_samples,))
      Nc : int
          RBFニューロン（クラスタ）の数

    Returns:
      W2 : numpy.ndarray
          RBFモデルの重み（形状: (Nc,)）
      B2 : float
          バイアス項
      Centers : numpy.ndarray
          RBFニューロンの中心（形状: (Nc, d)）
      Spreads : numpy.ndarray
          RBFモデルの幅（形状: (Nc,)）
    """
    X = np.array(SamIn)
    y = np.array(SamOut)
    num_samples, d = X.shape

    ClusterNum = Nc
    Overlap = 1.0

    # 初期中心はサンプルからランダムに選択
    indices = np.random.permutation(num_samples)
    Centers = X[indices[:ClusterNum], :].copy()

    max_iter = 50
    for iter in range(max_iter):
        NumberInClusters = np.zeros(ClusterNum, dtype=int)
        ClusterIndices = [[] for _ in range(ClusterNum)]
        for i in range(num_samples):
            distances = np.linalg.norm(Centers - X[i, :], axis=1)
            closest = np.argmin(distances)
            NumberInClusters[closest] += 1
            ClusterIndices[closest].append(i)

        OldCenters = Centers.copy()
        for i in range(ClusterNum):
            if NumberInClusters[i] != 0:
                Centers[i, :] = np.mean(X[ClusterIndices[i], :], axis=0)
            else:
                Centers[i, :] = X[np.random.randint(num_samples), :]

        if np.allclose(Centers, OldCenters):
            break

    # 中心間距離の計算
    pairwise_dists = np.linalg.norm(
        Centers[:, np.newaxis, :] - Centers[np.newaxis, :, :], axis=2
    )
    max_dist = np.max(pairwise_dists)
    np.fill_diagonal(pairwise_dists, max_dist + 1)
    pairwise_dists[pairwise_dists == 0] = 0.000001
    Spreads = Overlap * np.min(pairwise_dists, axis=1)

    distances = cdist(Centers, X)
    spreads_matrix = Spreads[:, np.newaxis]
    HiddenUnitOut = radbas(distances / spreads_matrix)
    HiddenUnitOutEx = np.vstack([HiddenUnitOut, np.ones((1, num_samples))])
    # HiddenUnitOutEx = np.vstack([HiddenUnitOut.T, np.ones(num_samples)])
    W2Ex = np.dot(y.reshape(1, -1), pinv(HiddenUnitOutEx))
    # W2Ex = np.dot(y, pinv(HiddenUnitOutEx.T))
    W2 = W2Ex[0, :ClusterNum]
    B2 = W2Ex[0, ClusterNum]

    return W2, B2, Centers, Spreads


def RBF_EnsembleUN(L, c, nc, T):
    """
    ブートストラップサンプリングによりRBFモデルのアンサンブルを構築

    Parameters:
      L : numpy.ndarray
          オフラインデータ (形状: (num_samples, c+1))
      c : int
          決定変数の数
      nc : int
          各RBFモデルのニューロン数
      T : int
          RBFモデルの数

    Returns:
      W : numpy.ndarray
          各RBFモデルの重み (形状: (T, nc))
      B : numpy.ndarray
          各RBFモデルのバイアス (形状: (T,))
      C_arr : numpy.ndarray
          各RBFモデルの中心 (形状: (T, nc, c))
      S_arr : numpy.ndarray
          各RBFモデルの幅 (形状: (T, nc))
    """
    t = 0.5
    num_samples = L.shape[0]

    W = np.zeros((T, nc))
    B = np.zeros(T)
    C_arr = np.zeros((T, nc, c))
    S_arr = np.zeros((T, nc))

    for i in range(T):
        mask = np.random.rand(num_samples) < t
        if np.sum(mask) == 0:
            mask[np.random.randint(num_samples)] = True
        L1 = L[mask, :]
        X = L1[:, :c]
        y = L1[:, c]
        W2, B2, Centers, Spreads = RBF(X, y, nc)
        W[i, :] = W2
        B[i] = B2
        C_arr[i, :, :] = Centers
        S_arr[i, :] = Spreads

    return W, B, C_arr, S_arr


def RBF_Ensemble_predictor(W, B, C_arr, S_arr, U, c):
    """
    サロゲートモデルエンサンブルによる予測

    Parameters:
      W : numpy.ndarray
          各RBFモデルの重み (形状: (num_models, num_neurons))
      B : numpy.ndarray
          各RBFモデルのバイアス (形状: (num_models,))
      C_arr : numpy.ndarray
          各RBFモデルの中心 (形状: (num_models, num_neurons, c))
      S_arr : numpy.ndarray
          各RBFモデルの幅 (形状: (num_models, num_neurons))
      U : numpy.ndarray
          テスト入力 (形状: (num_samples, c))
      c : int
          決定変数の数

    Returns:
      Y : numpy.ndarray
          各モデルの予測値 (形状: (num_samples, num_models))
    """
    num_models = W.shape[0]
    num_samples = U.shape[0]
    Y = np.zeros((num_samples, num_models))
    for i in range(num_models):
        Y[:, i] = RBF_predictor(W[i, :], B[i], C_arr[i, :, :], S_arr[i, :], U)
    return Y


def RBF_predictor(W2, B2, Centers, Spreads, TestSamIn):
    """
    単一RBF予測器

    Parameters:
      W2 : numpy.ndarray
          RBFモデルの重み (形状: (num_neurons,))
      B2 : float
          バイアス項
      Centers : numpy.ndarray
          RBFニューロンの中心 (形状: (num_neurons, d))
      Spreads : numpy.ndarray
          RBFニューロンの幅 (形状: (num_neurons,))
      TestSamIn : numpy.ndarray
          テスト入力 (形状: (num_samples, d))

    Returns:
      TestNNOut : numpy.ndarray
          予測出力 (形状: (num_samples,))
    """
    distances = cdist(Centers, TestSamIn)
    normalized = distances / Spreads[:, np.newaxis]
    HiddenOut = radbas(normalized)
    TestNNOut = np.dot(W2, HiddenOut) + B2
    return TestNNOut
