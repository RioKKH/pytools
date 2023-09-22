#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from random_forest import RandomForest


def main():
    # MNISTデータセットのロード
    mnist = datasets.fetch_openml('mnist_784', version=1)
    X = mnist.data
    y = mnist.target

    # データセットの分割
    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.2, random_state=42)

    # ランダムフォレストモデルのインスタンス化と訓練
    rf_model = RandomForest(n_trees=10, max_depth=10)
    rf_model.fit(X_train, y_train)

    # テストデータの予測
    y_pred = rf_model.predict(X_test)

    # 予測の評価
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

