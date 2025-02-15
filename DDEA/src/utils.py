#!/usr/bin/env python

import numpy as np


def lhs_design(n, d):
    """
    ラテンハイパーキューブサンプリングを生成

    Parameters:
      n : int
          サンプル数
      d : int
          次元数

    Returns:
      samples : numpy.ndarray
          形状 (n, d) のサンプル。各要素は [0,1] の値。
    """
    result = np.zeros((n, d))
    for i in range(d):
        perm = np.random.permutation(n) + 1
        result[:, i] = (perm - np.random.rand(n)) / n
    return result


def radbas(x):
    """
    放射基底関数 (radial basis function) の計算

    Parameters:
      x : numpy.ndarray
          入力配列

    Returns:
      numpy.ndarray
          各要素に対して exp(-x^2) を計算
    """
    return np.exp(-np.power(x, 2))


def cdist(A, B):
    """
    2つの配列間のユークリッド距離を計算

    Parameters:
      A : numpy.ndarray
          形状 (n, d)
      B : numpy.ndarray
          形状 (m, d)

    Returns:
      distances : numpy.ndarray
          形状 (n, m) の距離行列。distances[i, j] は A[i] と B[j] の距離。
    """
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))
