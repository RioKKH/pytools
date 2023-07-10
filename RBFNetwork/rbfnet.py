#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans


# RBF関数
def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x - c)**2)


# RBFネットワーク
class RBFNet:

    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k                  # number of RBF centers
        self.lr = lr                # learning rate
        self.epochs = epochs        # number of training epochs
        self.rbf = rbf              # RBF function
        self.inferStds = inferStds  # whether to infer standard deviations

        self.w = np.random.randn(k) # weights
        self.b = np.random.randn(1) # bias
        self.centers = None         # centers
        self.stds = None            # standard deviations

    def fit(self, X, y):
        # Use k-means to determine the clusters
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        if self.inferStds:
            # Infer standard deviations from the data
            dists = np.array([np.abs(c1 - c2)
                              for c1 in self.centers for c2 in self.centers])
            dMax = np.max(dists)
            self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)
        else:
            # Use a fixed standard deviation
            self.stds = np.repeat(1.0, self.k)

        # Training
        # 重みの学習
        # 重みは勾配降下法を用いて学習している
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # Forward pass
                a = np.array([self.rbf(X[i], c, s)
                              for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                #loss = (y[i] - F).flatten() ** 2
                #print('Loss: {loss[0]:.2f}')

                # Backward pass
                error = -(y[i] - F).flatten()

                # Update weights and bias
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


def f(x):
    return x[0]**2 + x[1]**2


# RBFネットワークの初期化
rbf_net = RBFNet(k=2, lr=0.01, epochs=100)

# データの生成
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = f(X)

# RBFネットワークの学習
rbf_net.fit(X, y)

# 遺伝的アルゴリズムで最適化
bounds = [(-5, 5)]
result = differential_evolution(lambda x: rbf_net.predict(np.array([x].reshape(-1, 2))), bounds)
#result = differential_evolution(lambda x: rbf_net.predict(np.array([x])), bounds)

# 最適解の表示
print(f"Global minimum: x = {result.x[0]:.3f}")

