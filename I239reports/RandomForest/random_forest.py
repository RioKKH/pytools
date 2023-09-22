#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from decision_tree import DecisionTree


class RandomForest:

    def __init__(self, n_trees=100, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """Build a forest of trees."""
        for _ in range(self.n_trees):
            # ブートストラップサンプリング
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample = X.iloc[indices] \
                if isinstance(X, pd.DataFrame) else X[indices]
            y_sample = y[indices]

            # 決定木の構築と訓練
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """Predict class for X."""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.argmax(np.bincount(predictions[:, i]))\
                         for i in range(predictions.shape[1])])

