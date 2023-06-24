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
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters)
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
