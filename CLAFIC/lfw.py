#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from clafic import CLAFIC


class FacesClassifier:

    def __init__(self):
        self.data = None
        self.images = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clafic = None
        self.y_pred = None
        self.accuracy = None
        self.class_names = None

    def load_data(self):
        faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        self.class_names = faces.target_names
        self.images = faces.images
        X = faces.data
        y = faces.target
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(X, y, test_size=0.2, random_state=42)

        n_features = X.shape[1]
        n_samples, h, w = self.images.shape
        target_names = faces.target_names
        n_classes = target_names.shape[0]
        print("Total dataset size:")
        print(f"n_samples: {n_samples}")
        print(f"n_features: {n_features}")
        print(f"h x w: {h} x {w}")
        print(f"n_classes: {n_classes}")

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

        # colors = ['r', 'g', 'b']
        for i, class_name in zip(self.clafic.classes, self.class_names):
        #for i, c, class_name in zip(self.clafic.classes, colors, self.class_names):
            eigvals = self.clafic.models[i][2]
            cum_var_exp = np.cumsum(eigvals) / np.sum(eigvals)
            ax.plot(range(1, len(cum_var_exp) + 1),
                    cum_var_exp,
                    marker='o',
                    alpha=0.8,
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
                                      #xticks_rotation="vertical")
                                      #display_labels=self.clafic.classes)
        disp.plot(include_values=True, cmap='viridis',
                  ax=None, xticks_rotation='vertical')
        plt.tight_layout()
        plt.show()

    def plot_images(self, indices):
        fig, axes = plt.subplots(1, len(indices), figsize=(10, 2))
        for ax, index in zip(axes, indices):
            ax.imshow(self.images[index], cmap='gray')
            ax.axis('off')
        plt.show()

    def plot_bases(self, n_bases):
        fig, axes = plt.subplots(1, len(self.class_names), figsize=(10, 2))
        for ax, (_, V, _) in zip(axes, self.clafic.models.values()):
            ax.imshow(V[:, 0].reshape(self.images[0].shape), cmap='gray')
            ax.axis('off')
        plt.show()

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.clafic, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.clafic = pickle.load(f)


def main(showdata=False, run_train=True, numdim=5):
    # Initialize model
    model = FacesClassifier()

    # Load data
    model.load_data()

    # Train and predict for each number of components
    if run_train:
        for n_components in range(1, numdim):
            print(f"progress of training: {n_components}")
            path = f"clafic_model_{n_components}.pkl"
            if not os.path.exists(path):
                model.train(n_components)
                model.save_model(path)

    accuracies = []
    for n_components in range(1, numdim):
        model.load_model(f"clafic_model_{n_components}.pkl")
        model.predict()
        accuracies.append(model.accuracy)

    # Plot accuracy for each number of compontents
    plt.plot(range(1, numdim), accuracies, marker='o')
    plt.title("Accuracy for Different Number of Dimensions")
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    #plt.xlim(0, 5)
    plt.grid(ls='dashed', color='gray', alpha=0.5)
    plt.show()

    # Plot cumulative explained variance for each class
    model.load_model(f"clafic_model_{numdim-1}.pkl")
    model.plot_cumulative_explained_variance()

    #model.plot_eigenvalues()

    # Display confusion matrix
    model.plot_confusion_matrix()

    # Plot some images
    model.plot_images([0, 100, 200])
    
    # Plot basis images
    model.plot_bases(n_bases=numdim)

    return model
