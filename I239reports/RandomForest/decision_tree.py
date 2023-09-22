#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from node import Node, find_best_split


class DecisionTree:

    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        """Build the decision tree."""
        # What does this set method do?
        # It returns a new set object, optionally with elements taken from iterable.
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        """Predict class for X."""
        # Xをnumpy配列に変換し、各サンプルが1次元配列になるようにする
        X = np.atleast_2d(X)
        return [self._predict(inputs) for inputs in X]
        #return [self._predict(inputs) for inputs in X]

    # _がついているがこれは外部から呼び出されることを想定していない
    # メソッドであることを示している。
    def _grow_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split."""
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini = 1 - sum((n / len(y)) ** 2 for n in num_samples_per_class),
            num_samples = len(y),
            num_samples_per_class = num_samples_per_class,
            predicted_class = predicted_class,
        )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            # 特徴量のランダムサブセットを選択する
            subset_size = int(np.sqrt(self.n_features_))
            subset_feature_indices = np.random.choice(self.n_features_,
                                                      subset_size,
                                                      #self.n_features_,
                                                      replace=False)

            # 最良の分割を見つける
            idx, thr = find_best_split(X[:, subset_feature_indices], y,
                                       self.n_features_, self.n_classes_)
            if idx is not None:
                # サブセットからのインデックスを元の特徴量のインデックスに変換する
                idx = subset_feature_indices[idx]

                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                # What does the ~ operator do?
                # It returns the complement of the input array, element-wise.
                # In this case, it returns the indices of the right side of the split.
                X_right, y_right = X[~indices_left], y[~indices_left]

                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

