#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from basemodel import BaseModel


class RandomForestModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.accuracy = None

    def load_data(self):
        # """CIFAR-10のデータセットをロードする"""
        # cifar10 = fetch_openml('CIFAR_10', version=1)
        # X = cifar10.data
        # y = cifar10.target

        # """MNISTのデータセットをロードする"""
        mnist = fetch_openml('mnist_784', version=1)
        X = mnist.data
        y = mnist.target

        # データセットを訓練用とテスト用に分割する
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(X, y, test_size=0.2, random_state=42)

    def save_model(self, filepath:str) -> None:
        """ランダムフォレストクラス分類器の評価結果を保存する"""
        print(f"Saving model to: {filepath}")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath:str) -> None:
        """ランダムフォレストクラス分類器の評価結果を読み込む"""
        print(f"Loading model from: {filepath}")
        self.model = joblib.load(filepath)

    def preprocess_data(self):
        """RandomForestでは特に前処理は行わない"""
        pass

    def build_model(self):
        """ランダムフォレオストクラス分類器のインスタンスを作成する"""
        self.model = RandomForestClassifier(n_estimators=100, 
                                            random_state=42,
                                            verbose=1)

    def train_model(self):
        """ランダムフォレストクラス分類器を訓練する"""
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """テストデータを用いてランダムフォレストクラス分類器を評価する"""
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Test Accuracy: {self.accuracy}")

    def plot_images(self, num_images=5):
        """ランダムに画像を選択してプロットする"""
        # ランダムに画像を選択する
        indices = np.random.choice(np.arange(len(self.X_train),
                                             num_images,
                                             replace=False))
        selected_images = self.X_train[indices]
        selected_labels = self.y_train[indices]

        # 選択した画像をプロットする
        plt.figure(figsize(10, num_images))
        for i, (img, label) in enumerate(zip(selected_images, seleted_labels)):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img.reshape(28, 28), cmap='gray')
            #plt.xticks([])
            #plt.yticks([])
            plt.axis('off')
            plt.title(f"Label: {label}")
        plt.tight_layout()
        plt.show()

    def plot_results(self):
        """ランダムフォレストクラス分類器の評価結果をグラフで表示する"""
        # 混同行列の計算
        cm = confusion_matrix(self.y_test, self.y_pred)
        # 各クラスの正答率の計算
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        # クラスラベル
        class_labels = np.unique(self.y_test.astype(int))

        plt.figure(figsize=(12, 5))

        # 正答率の棒グラフを表示
        plt.subplot(1, 2, 1)
        bars = plt.bar(class_labels, class_accuracy)
        plt.xlabel('Class label')
        plt.ylabel('Accuracy')
        plt.title('Class-wise accuracy of mnist dataset')
        plt.xticks(class_labels)
        plt.ylim(0, 1)

        # 各バーに対応する数値を表示
        for bar, acc in zip(bars, class_accuracy):
            plt.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() - 0.05,
                     f"{acc:.2f}",
                     ha='center',
                     va='top',
                     color='white',
                     fontsize=12)

        # 混同行列のヒートマップを表示
        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix of mnist dataset')

        plt.tight_layout()
        plt.show()


    def run(self):
        """モデルの学習から評価までの一連の処理を実行する"""
        self.load_data()
        self.preprocess_data()
        self.build_model()
        self.train_model()
        self.evaluate_model()
        filepath = "random_forest_model.joblib"
        self.save_model(filepath)
        self.plot_results()


if __name__ == '__main__':
    model = RandomForestModel()
    model.load_data()
    model.preprocess_data()
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save_results()
    model.plot_results()

