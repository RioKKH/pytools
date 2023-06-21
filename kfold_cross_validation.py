#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self):
        self.X = None
        self.y = None

    def load_data(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target

    def get_data(self):
        return self.X, self.y


class CrossValidator:
    def __init__(self, model, k = 5):
        self.model = model
        self.k = k
        self.kf = KFold(n_splits=k, shuffle=True, random_state=42)

        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # 正解率
        # モデルが正しく分類したデータの割合を示す指標。具体的には正解した
        # データの数を全データ数で割った値になる。
        # 全体の予測の中で正しく予測された割合を示す。データのバランスが
        # 取れている場合や、陽性と陰性の重要性が均等である場合に適している。
        # ただし、クラスのバランスが偏っている場合や、クラスの重要度が異なる
        # 場合には正確な評価指標とは言えない
        # 正解率 = (真陽性 + 真陰性) / (真陽性 + 偽陽性 + 真陰性 + 偽陽性)
        accuracy = accuracy_score(y_test, y_pred)
        # 適合率
        # 適合率は陽性と予測されたデータのうち、実際に陽性であるデータの
        # 割合を示す指標である。つまりモデルが要請と予測したデータのうち、
        # 実際に陽性である割合の事。適合率は偽陽性を最小化することを重視
        # する場合に有用な指標となる。
        # 例えば不正検出の場合など。    
        # 適合率 = 真陽性 / (真陽性 + 偽陽性)
        precision = precision_score(y_test, y_pred, average='weighted')
        # 再現率
        # 再現率は、実際に陽性であるデータのうち、モデルが正しく陽性と予測できた
        # 割合を示す指標である。つまり、全陽性データのうち、モデルが陽性と予測
        # した割合です。再現率は偽陰性を最小化することを重視する場合に有用。
        # 陽性クラスを見逃さない事を重視する場合に有用。例えば、病気の検出などで
        # 偽陰性を最小化する必要がある場合に重要な指標である
        # 再現率 = 真陽性 / (真陽性 + 偽陰性)
        recall = recall_score(y_test, y_pred, average='weighted')
        # F1スコア
        # F1スコアは、precision(適合率)とrecall(再現率)のバランスを取った指標。
        # 適合率と再現率の調和平均を計算することで求められる。
        # F1スコアは適合率と再現率の両方を高く保つ事を目指す場合に有用。
        # つまり偽陽性を最小化し(--> 適合率)、偽陰性を最小化する(--> 再現率)
        # 均等な重みをもつ場合に有用である
        # F1スコア = 2 * (適合率 * 再現率) / (適合率 + 再現率)
        f1 = f1_score(y_test, y_pred, average='weighted')

        self.accuracies.append(accuracy)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)

        return accuracy, precision, recall, f1

    def run_cross_validation(self, X, y):
        fold = 1
        for train_index, test_index in self.kf.split(X):
            print("Fold:", fold)
            print("Training samples:", len(train_index))
            print("Testing samples:", len(test_index))

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            accuracy, precision, recall, f1 =\
                self.train_and_evaluate(X_train, X_test, y_train, y_test)

            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score", f1)

            fold += 1
            print("------------------------------")

    def plot_evaluation_scores(self):
        fig, ax = plt.subplots()
        ax.plot(range(1, self.k + 1), self.accuracies, marker='o', label='Accuracy')
        ax.plot(range(1, self.k + 1), self.precisions, marker='o', label='Precision')
        ax.plot(range(1, self.k + 1), self.recalls,    marker='o', label='Recall')
        ax.plot(range(1, self.k + 1), self.f1_scores,  marker='o', label='F1 Score')

        ax.set_xlabel("Fold")
        ax.set_ylabel("Score")
        ax.set_title("Evaluation Scores")
        ax.legend()
        plt.show()

    def plot_classification_results(self, X, y):
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        unique_classes = np.unique(y)

        colors = ['r', 'g', 'b'] # クラスごとの色
        markers = ['o', 's', '^'] # クラスごとのマーカー

        for cls in unique_classes:
            class_indices = np.where(y == cls)
            plt.scatter(X[class_indices, 0], X[class_indices, 1],
                        color=colors[cls], marker=markers[cls],
                        label=f'Class {cls}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Classfication Results')
            plt.legend()
            plt.show()



if __name__ == '__main__':
    # データの処理と準備
    data_processor = DataProcessor()
    data_processor.load_data()
    X, y = data_processor.get_data()

    # モデルの設定
    model = KNeighborsClassifier(n_neighbors=3)

    # CrossValidatorのインスタンス化と実行
    cross_validator = CrossValidator(model, k=5)
    cross_validator.run_cross_validation(X, y)

    # 評価指標のプロット
    cross_validator.plot_evaluation_scores()

    # 分類結果のプロット
    cross_validator.plot_classification_results(X, y)
