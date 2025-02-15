#!/usr/bin/env python

import numpy as np
from .rbf import RBF_Ensemble_predictor


def SelectModels(W, B, C_arr, S_arr, Xb, c, Q):
    """
    現在の最良個体に対する予測結果に基づき、サロゲートモデル群から一部モデルを選択する

    Parameters:
      W : numpy.ndarray
          全RBFモデルの重み (形状: (T, num_neurons))
      B : numpy.ndarray
          全RBFモデルのバイアス (形状: (T,))
      C_arr : numpy.ndarray
          全RBFモデルの中心 (形状: (T, num_neurons, c))
      S_arr : numpy.ndarray
          全RBFモデルの幅 (形状: (T, num_neurons))
      Xb : numpy.ndarray
          最良個体 (形状: (1, c))
      c : int
          決定変数の数
      Q : int
          選択するモデル数

    Returns:
      selected_indices : numpy.ndarray
          選択されたモデルのインデックス (形状: (Q,))
    """
    # 最良個体に対する各モデルの予測値（Xbは1サンプル）
    Y = RBF_Ensemble_predictor(W, B, C_arr, S_arr, Xb, c)
    Y = Y.flatten()
    T = Y.shape[0]
    sorted_indices = np.argsort(Y)
    selected_indices = []
    for i in range(Q):
        idx = int(np.ceil((i + 1) * T / Q)) - 1
        selected_indices.append(sorted_indices[idx])
    return np.array(selected_indices)
