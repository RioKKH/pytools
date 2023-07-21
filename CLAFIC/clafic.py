#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from scipy.linalg import svd


class CLAFIC:

    def __init__(self, n_components:int=2) -> None:
        self.n_components = n_components
        self.models = {}

    def _KL_expansion(self, X):
        # Compute mean
        X_mean = np.mean(X, axis=0)
        # Cntering data
        X_centered = X - X_mean
        # Compute covariance matrix
        C = np.cov(X_centered.T)
        # Compute eigenvalues and eigenvectors of the covariance matrix
        #eigvals, S, Vt = svd(C)
        # Compute eivenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(C)
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Select top n_components eigenvectors
        #V = Vt.T[:, :self.n_components]
        V = eigvecs[:, :self.n_components]
        # Return the mean and principal components and their eigenvalues
        return X_mean, V, eigvals[:self.n_components]

    def fit(self, X, y):
        """
        各クラスのデータに対してKL展開を実行し、その結果
        (平均と主成分, 固有値)を保存する。
        """
        self.classes = np.unique(y)
        for i in self.classes:
            X_i = X[y == i]
            self.models[i] = self._KL_expansion(X_i)

    def set_n_components(self, n_components):
        """
        部分空間の次元数を変更する際に用いるメソッド
        """
        self.n_components = n_components
        self.fit(self.X_train, self.y_train)

    def _project(self, x, model):
        """
        与えられたデータ点を指定された部分空間(平均と主成分)に射影する。
        射影は、データ点を部分空間の基底ベクトルに直交射影した後、平均を
        加える事で実現される。
        """
        X_mean, V, _ = model
        X_centered = x - X_mean
        x_projected = X_mean + V @ (V.T @ X_centered)
        return x_projected

    def predict(self, X):
        """
        各クラスの部分空間に対してデータ点を射影し、射影後の誤差
        (データ点と射影点のユークリッド距離)が最小となるクラスを選び出す。
        このクラスが予測クラスとなる。
        """
        y_pred = []
        for x in X:
            min_dist = float('inf')
            best_class = None
            for i in self.classes:
                x_projected = self._project(x, self.models[i])
                dist = np.linalg.norm(x - x_projected)
                if dist < min_dist:
                    min_dist = dist
                    best_class = i
            y_pred.append(best_class)
        return np.array(y_pred)
