#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from clafic import CLAFIC


class IrisClassifier:

    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clafic = None
        self.y_pred = None
        self.accuracy = None
        self.class_names = None
        #self.n_components = n_components
        #self.clafic = CLAFIC(n_components)

    def load_data(self):
        iris = load_iris()
        self.class_names = iris.target_names
        X = iris.data
        y = iris.target
        self.data = pd.DataFrame(data=np.c_[X, y],
                                 columns=iris.feature_names + ['class'])
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(X, y, test_size=0.2, random_state=42)
        #self.classes = iris.target_names

    def train(self, n_components):
        self.clafic = CLAFIC(n_components=n_components)
        self.clafic.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.clafic.predict(self.X_test)
        self.accuracy = (self.y_pred == self.y_test).mean()
        print(f"Accuracy: {self.accuracy}")

    def plot_cumulative_explained_variance(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(True)
        ax.set_title("Cumulative explained variance by number of dimensions")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_xlabel("Number of dimensions")

        colors = ['r', 'g', 'b']
        for i, c, class_name in zip(self.clafic.classes, colors, self.class_names):
            eigvals = self.clafic.models[i][2]
            cum_var_exp = np.cumsum(eigvals) / np.sum(eigvals)
            ax.plot(range(1, len(cum_var_exp) + 1),
                    cum_var_exp,
                    marker='o',
                    alpha=0.8,
                    color=c,
                    label=class_name)

        plt.ylim(0, 1.1)
        #plt.xlim(0, 5)
        plt.grid(ls='dashed', color='gray', alpha=0.5)
        ax.legend()
        plt.show()

    def plot_eigenvalues(self):
        """
        各クラスの部分空間での固有値（主成分の重要度）をプロットする。
        これにより、各クラスの部分空間がどの程度の情報を含んでいるか
        （どの程度の分散を説明しているか）を視覚的に理解することが出来る。
        """
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        for i, model in self.clafic.models.items():
            X_mean, V, eigvals = model
            axes[i].plot(eigvals, label=f"Class {self.classes[i]}")
            axes[i].set_title(f"Eigenvalues for Each Class {self.classes[i]}")
            axes[i].set_xlabel("Component")
            axes[i].set_ylabel("Eigenvalue")
            axes[i].legend()
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        """
        混同行列(confusion matrix)をプロットする。混同行列は、モデルの分類結果を
        詳細に理解するためのツールで、各行と各列が真のクラスと予測クラスに対応
        している。対角線上の値は正しく分類されたサンプル数を示し、非対角線上の
        値はご分類されたサンプル数を示す。
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.class_names)
                                      #display_labels=self.clafic.classes)
        disp.plot(include_values=True, cmap='viridis',
                  ax=None, xticks_rotation='horizontal')
        plt.show()

    def view_data(self) -> pd.DataFrame:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = [iris.target_names[i] for i in iris.target]
        return df


def main(showdata=False) -> pd.DataFrame:
    # Initialize model
    model = IrisClassifier()

    # Load data
    model.load_data()

    # Train and predict for each number of components
    accuracies = []
    for n_components in range(1, 5):
        model.train(n_components)
        model.predict()
        accuracies.append(model.accuracy)

    # Plot accuracy for each number of compontents
    plt.plot(range(1, 5), accuracies, marker='o')
    plt.title("Accuracy for Different Number of Dimensions")
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    #plt.xlim(0, 5)
    plt.grid(ls='dashed', color='gray', alpha=0.5)
    plt.show()

    # Plot cumulative explained variance for each class
    model.train(1)
    model.plot_cumulative_explained_variance()

    #model.plot_eigenvalues()

    # Display confusion matrix
    model.plot_confusion_matrix()

    # Display data
    df = model.view_data()
    if showdata:
        print(df.head())
    else:
        return df

