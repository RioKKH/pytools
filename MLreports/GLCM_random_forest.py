#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from tqdm.notebook import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.feature.texture import graycomatrix, graycoprops

from random_forest import RandomForestModel


class GLCMRandomForestModel(RandomForestModel):

    def __init__(self, n_trials=100):
        super().__init__()
        self.n_trials = n_trials


    def load_glcm_features(self):
        self.X_train_features = np.load('glcm_x_train_features.npy')
        self.X_test_features = np.load('glcm_x_test_features.npy')


    def compute_features(self):
        """GLCM特徴量を計算する"""
        X_train_images = self.X_train.to_numpy().reshape(-1, 28, 28)
        X_test_images = self.X_test.to_numpy().reshape(-1, 28, 28)

        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        properties = ['contrast', 'dissimilarity', 'energy', 
                      'homogeneity', 'ASM', 'correlation']

        self.X_train_features\
            = self._compute_glcm_features(X_train_images,
                                          distances, angles, properties)

        train_features = [feature.ravel() for feature in self.X_train_features]
        np.save('glcm_x_train_features.npy', train_features)

        self.X_test_features\
            = self._compute_glcm_features(X_test_images,
                                          distances, angles, properties)

        test_features = [feature.ravel() for feature in self.X_test_features]
        np.save('glcm_x_test_features.npy', test_features)


    def _compute_glcm_features(self, images, distances, angles, properties):
        features = []
        # ループをtqdmでラップして進捗バーを表示する
        for i, image in tqdm(enumerate(images),
                             total=len(images), desc='Computing GLCM features'):
            image = image.astype(np.uint8)
            glcm = graycomatrix(image,
                                distances=distances,
                                angles=angles,
                                symmetric=True,
                                normed=True)

            # 各特徴量の平均値を計算する
            # rabel()を用いて、glcmの各特徴量を1次元配列に変換する
            feature = [graycoprops(glcm, prop).ravel() for prop in properties]
            #feature = [graycoprops(glcm, prop).ravel().mean() for prop in properties]
            features.append(feature)
        return np.array(features)


    def build_model(self):
        """RandomForestモデルの構築する"""
        self.model = RandomForestClassifier(n_estimators=100, 
                                            random_state=42,
                                            verbose=1)

    def train_model(self):
        """GLCM特徴量を用いてモデルの学習を行う"""
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


    def run(self, should_compute=False):
        """モデルの学習から評価までの一連の処理を実行する"""
        self.load_data()
        if should_compute:
            self.compute_features()
        else:
            self.load_glcm_features()

        self.build_model()
        self.train_model()
        self.evaluate_model()
        filepath = "random_forest_model_glcm.joblib"
        self.save_model(filepath)
        self.plot_results()

