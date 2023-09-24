#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Node:

    def __init__(self,
                 gini, num_samples,
                 num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threashold = 0
        self.left = None
        self.right = None

def find_best_split(X, y, num_samples, num_classes):
    """Find the best split for a node."""
    # Initialize variables
    best_gini = 1.0 # the smallest gini value
    best_idx = None # the best feature index
    best_thr = None # the best threashold vlaue

    # Compute the Gini impurity of the current node.
    m = len(y)
    if m <= 1:
        return None, None

    num_parent = [np.sum(y == c) for c in range(num_classes)]
    best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)

    # Loop through all features and compute the Gini impurity of the split.
    for idx in range(X.shape[1]):
        thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
        num_left = [0] * num_classes
        num_right = num_parent.copy()

        for i in range(1, m):
            # move one sample from right to left
            c = classes[i - 1]
            # increase the number of samples on the left and decrease the number of samples on the right
            num_left[c] += 1
            num_right[c] -= 1
            # calculate the Gini impurity
            gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(num_classes))
            gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(num_classes))
            # the Gini impurity of a split is the weighted average of the Gini impurity of the children
            gini = (i * gini_left + (m - i) * gini_right) / m

            if thresholds[i] == thresholds[i - 1]:
                continue

            if gini < best_gini:
                best_gini = gini
                best_idx = idx
                best_thr = (thresholds[i] + thresholds[i - 1]) / 2

    return best_idx, best_thr
