#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self):
        self.data = None
        self.target = None

    def load_data(self):
        iris = load_iris()
        self.data = iris.data
        self.target = iris.target


class KMeansModel:
    """
    scikit-learnのKMeansクラスはデフォルトでk-means++の初期化方法を採用している。
    k-means++は初期値選択方法の一つ。k-meansアルゴリズムは、ランダムな初期値に
    よってクラスタリングの結果が大きく変わる可能性があるので、その初期値選択は
    重要。k-means++はその初期値選択問題を改善するために提案された。

    1. データセットからランダムに１つのデータポイントを選び、最初のクラスタ
    センター（重心）とする。
    2. 他の各データポイントについて、既存のクラスターセンターまでの距離を計算し、
    それを二乗する。これをD(x)と呼ぶ。
    3. D(x) を元に、各データポイントが次のクラスタセンターに選ばれる確率を計算
    する。具体的には、D(x)が大きいデータポイントほど次のクラスタセンターに
    選ばれる確率が高くなる。
    4. 確率に基づいてランダムに一つのデータポイントを選び、次のクラスタセンター
    とする。
    5. 指定されたクラスタ数kに達するまで、2~4の手順を繰り返す

    この初期化手順により、k-means++ ではクラスタセンターがデータセット上に
    均等に分布するようになり、k-meansアルゴリズムの結果の安定性と効率性が向上
    する。
    """
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters)
        # もしk-means++を使用することを明示したい場合は、以下のようにinit
        # パラメータを指定すればよい
        # self.model = KMeans(n_clusters=n_clusters, init='k-means++')
        self.predictions = None

    def fit_predict(self, data):
        self.predictions = self.model.fit_predict(data)


class Pipeline:
    def __init__(self, loader, model):
        self.loader = loader
        self.model = model

    def run(self):
        self.loader.load_data()
        self.model.fit_predict(self.loader.data)


class Visualizer:
    @staticmethod
    def visualize(data, predictions):
        plt.scatter(data[:, 0], data[:, 1], c=predictions)
        plt.show()


def main():
    data_loader = DataLoader()
    model = KMeansModel()
    pipeline = Pipeline(data_loader, model)
    pipeline.run()
    Visualizer.visualize(data_loader.data, model.predictions)


if __name__ == '__main__':
    main()
