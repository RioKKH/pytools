#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from basemodel import BaseModel
import joblib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class SVMModel(BaseModel):

    def __init__(self, kernel:str='rbf', C:float=1.0, gamma:str='auto'):
        super().__init__()
        self.kernel = kernel # カーネルタイプ
        self.C = C           # 正則化パラメータ
        self.gamma = gamma

    def load_data(self):
        train_size = 5000
        test_size  = 1000
        mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
        X = mnist.data
        y = mnist.target

        self.X_train, self.X_test, self.y_train, self.y_test\
            = train_test_split(X, y, 
                               test_size=0.2, random_state=42, stratify=y)
            #= train_test_split(X, y, 
            #                   train_size = train_size,
            #                   test_size = test_size,
            #                   random_state=42, stratify=y)

        self.X_train = self.X_train / 255
        self.X_test  = self.X_test  / 255

    def save_model(self, filepath:str) -> None:
        print(f"Saving model to {filepath}")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath:str) -> None:
        print(f"Loading model from {filepath}")
        self.model = joblib.load(filepath)

    def preprocess_data(self):
        pass

    def build_model(self):
        # gamma = 'auto' is used to avoid the warning message
        # what does gamma mean?
        # gamma is a parameter for non linear hyperplanes. 
        # The higher the gamma value it tries to exactly fit the training data set
        # What does C mean?
        # C is the penalty parameter of the error term. 
        # It controls the trade off between smooth decision boundary
        # and classifying the training points correctly.
        self.model = SVC(kernel=self.kernel,
                         C=self.C,
                         gamma=self.gamma)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Test Accuracy: {self.accuracy}")


    def objective(self, trial):
        C = trial.suggest_loguniform('C', 1e-10, 1e10)
        self.model = SVC(kernel=self.kernel, C=C, verbose=True)

        # Model training
        self.model.fit(self.X_train, self.y_train)

        # Model evaluation
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def plot_results(self):
        """SVM分類器の評価結果をグラフで表示する"""
        # 混同行列の計算
        cm = confusion_matrix(self.y_test, self.y_pred)
        # 各クラスの正答率の計算
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        # クラスラベル
        class_labels = np.unique(self.y_test.astype(int))

        plt.figure(figsize=(12, 5))

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

        # 混同行列のヒートマップを表示する
        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix of mnist dataset')

        plt.tight_layout()
        plt.show()


    def run(self):
        self.load_data()
        self.preprocess_data()
        #study = optuna.create_study(direction='maximize')
        #study.optimize(self.objective, n_trials=100)
        #print("Number of finished trials: ", len(study.trials))
        #print("Best trial:")
        #trial = study.best_trial
        #print("  Value: ", trial.value)
        #print("  Params: ")
        #for key, value in trial.params.items():
        #    print(f"    {key}: {value}")

        self.build_model()
        self.train_model()
        self.evaluate_model()
        filepath = f"svm_{self.kernel}_{self.C}_model.joblib"
        self.save_model(filepath)
        self.plot_results()
