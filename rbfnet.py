#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize


# RBF関数
def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x - c)**2)


# RBFネットワーク
class RBFNet:

    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)
        self.centers = None
        self.stds = None

    def fit(self, X, y):
        # RBFネットワークの中心と標準偏差はk-means法を用いて決定している。
        if self.inferStds:
            self.centers, self.stds = kmeans(X, self.k)
        else:
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        # 重みの学習
        # 重みは勾配降下法を用いて学習している
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # フォワードパス
                a = np.array([self.rbf(X[i], c, s)
                              for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2
                print('Loss: {loss[0]:.2f}')

                # バックワードパス
                error = -(y[i] - F).flatten()

                # 重みの更新
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)
