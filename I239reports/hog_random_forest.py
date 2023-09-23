#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from random_forest import RandomForestModel

class HOGRandomForestModel(RandomForestModel):

    def __init__(self):
        super().__init__()

    def compute_features(self):
        """HOG特徴量を計算する"""
        #訓練データのHOG特徴量を計算する
        self.X_train_features\
            = np.array([hog(img.reshape(28, 28)) for img in self.X_train])

        # テストデータのHOG特徴量を計算する
        self.X_test_features\
            = np.array([hog(img.reshape(28, 28)) for img in self.X_test])

    def build_model(self):
        """RandomForestモデルの構築する"""
        self.model = RandomForestClassifier(random_state=42)

    def train_model(self):
        """HOG特徴量を用いてモデルの学習を行う"""
        self.model.fit(self.X_train_features, self.y_train)

    def evaluate_model(self):
        """モデルの評価を行う"""
        self.y_pred = self.model.predict(self.X_test_features)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

    def run(self):
        """モデルの学習から評価までの一連の処理を実行する"""
        self.load_data()
        self.compute_features()
        self.build_model()
        self.train_model()
        self.evaluate_model()
        filepath = "random_forest_model_hog.joblib"
        self.save_model(filepath)
        self.plot_result()

