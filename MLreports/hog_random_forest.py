#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import optuna

from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from random_forest import RandomForestModel


class HOGRandomForestModel(RandomForestModel):

    def __init__(self, n_trials=50):
        super().__init__()
        self.n_trials = n_trials

    def compute_features(self):
        """HOG特徴量を計算する"""
        #訓練データのHOG特徴量を計算する
        self.X_train_features\
            = self.X_train.reshape(-1, 28, 28).astype(np.uint8)
        self.X_test_features\
            = self.X_test.reshape(-1, 28, 28).astype(np.uint8)

        #self.X_train_features\
        #    = np.array([hog(img.reshape(28, 28))
        #                for img in self.X_train.to_numpy()])

        ## テストデータのHOG特徴量を計算する
        #self.X_test_features\
        #    = np.array([hog(img.reshape(28, 28))
        #                for img in self.X_test.to_numpy()])

    def compute_features_with_params(self, params):
        """
        Optunaによって探索されたハイパーパラメータを用いて HOG特徴量を計算する
        """
        pixels_per_cell = params['pixels_per_cell']
        cells_per_block = params['cells_per_block']
        orientations = params['orientations']

        # トレーニングデータとテストデータのHOG特徴量を計算する
        self.X_train_features\
            = np.array([hog(img.reshape(28, 28),
                            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                            cells_per_block=(cells_per_block, cells_per_block),
                            orientations=orientations)
                        for img in self.X_train.to_numpy()])

        self.X_test_features\
            = np.array([hog(img.reshape(28, 28),
                            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                            cells_per_block=(cells_per_block, cells_per_block),
                            orientations=orientations)
                        for img in self.X_test.to_numpy()])

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

    def plot_images(self, num_images=5):
        """ランダムに画像を選択してプロットする"""
        #ランダムに画像を選択する
        indices = np.random.choice(len(self.X_train),
                                   num_images,
                                   replace=False)
        selected_images = self.X_train.to_numpy()[indices]
        selected_labels = self.y_train.to_numpy()[indices]

        #選択した画像をプロットする
        plt.figure(figsize=(20, num_images * 2))
        for i, (img, label) in enumerate(zip(selected_images, selected_labels)):
            features, hog_img = hog(img.reshape(28, 28), visualize=True)

            plt.subplot(2, num_images, i + 1)
            plt.imshow(img.reshape(28, 28), cmap='binary')

            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(hog_img.reshape(28, 28), cmap='binary')
            plt.axis('off')
            plt.title('Label: ' + str(label))
        plt.tight_layout()
        plt.show()

    def objectives(self, trial):

        img_size = 28
        max_pixels_per_cell = img_size
        max_cells_per_block = img_size\
            // trial.suggest_int('pixels_per_cell', 1, max_pixels_per_cell)
        # HOGのハイパーパラメータを設定する
        pixels_per_cell = trial.suggest_int('pixels_per_cell', 1, max_pixels_per_cell)
        # cells_per_blockが意味するのは、セルの分割数
        cells_per_block = trial.suggest_int('cells_per_block', 1, max_cells_per_block)
        # orientationsが意味するのは、特徴量の方向の分割数
        orientations = trial.suggest_int('orientations', 1, 16)

        # Startified K-Fold Cross Validationでモデルの評価を行う
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        accuracies = []
        for train_index, valid_index in skf.split(self.X_train_val, self.y_train_val):
            X_train = self.X_train_val.to_numpy()[train_index]
            X_valid = self.X_train_val.to_numpy()[valid_index]
            y_train = self.y_train_val.to_numpy()[train_index]
            y_valid = self.y_train_val.to_numpy()[valid_index]

            # HOG特徴量の計算
            X_train_features = np.array([hog(img.reshape(28, 28),
                                             pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                             cells_per_block=(cells_per_block, cells_per_block),
                                             orientations=orientations)
                                         for img in X_train])

            X_valid_features = np.array([hog(img.reshape(28, 28),
                                             pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                             cells_per_block=(cells_per_block, cells_per_block),
                                             orientations=orientations)
                                         for img in X_valid])

            # モデルの構築、学習、評価
            self.build_model()
            self.model.fit(X_train_features, y_train)
            y_pred = self.model.predict(X_valid_features)
            accuracies.append(accuracy_score(y_valid, y_pred))

        # 評価結果の平均を返す
        return 1 - np.mean(accuracies)

    def optimize_hyperparams(self):
        # Optunaでハイパーパラメータの最適化を行う
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objectives, n_trials=self.n_trials)

        print("Best trial:")
        trial = self.study.best_trial
        print("  Value: ", 1.0 - trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"{key}:{value}")


    def run(self):
        """モデルの学習から評価までの一連の処理を実行する"""
        print(self.n_trials)
        self.load_data()

        # データをトレーニングデータとテストデータに分割する
        self.X_train_val, self.X_valid, self.y_train_val, self.y_valid\
            = train_test_split(self.X_train, self.y_train,
                               test_size=0.2, stratify=self.y_train,
                               random_state=42)

        #self.optimize_hyperparams()
        #self.compute_features_with_params(self.study.best_params)
        self.compute_features_with_params({"pixels_per_cell": 7, "cells_per_block": 4, "orientations": 6})
        self.build_model()
        self.train_model()
        self.evaluate_model()
        filepath = "random_forest_model_hog.joblib"
        self.save_model(filepath)
        self.plot_results()

