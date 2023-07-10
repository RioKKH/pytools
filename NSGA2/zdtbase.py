#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from nsga2.problem import Problem
from nsga2.evolution import Evolution


class ZDTBase(ABC):
    """
    抽象基底クラス(Abstract Base Class)と抽象メソッドを用いて、
    C++の純粋仮想関数に相当する機能を実装する。
    """
    @abstractmethod
    def f1(self):
        pass

    @abstractmethod
    def f2(self):
        pass

    @abstractmethod
    def pareto_optimal_front(self):
        pass

    def run_evolution(self):
        # 問題の定義
        problem = Problem(num_of_variables=30,
                          objectives=[self.f1, self.f2],
                          variables_range=[(0, 1)],
                          same_range=True,
                          expand=False)

        # NSGA-IIの進化プロセスの初期化
        evo = Evolution(problem)
        # 進化プロセスの実行
        # 各世代の最良フロントを保存する
        self.best_fronts = evo.evolve()

    def save(self, filename):
        # best_frontsをpickle形式で保存する
        with open(filename, 'wb') as f:
            pickle.dump(self.best_fronts, f)

    
    def load(self, filename):
        # pickle形式のデータを読み込む
        with open(filename, 'rb') as f:
            self.best_fronts = pickle.load(f)

    def plot(self, generation, 
             xmin=-0.2, xmax=1.2,
             ymin=-0.2, ymax=1.2,
             xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
             yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):

        # 指定された世代の最良のフロントを取得
        best_front = self.best_fronts[generation]
        func = [i.objectives for i in best_front]

        # 目的関数1と目的関数2の値をそれぞれ取得
        function1 = [i[0] for i in func]
        function2 = [i[1] for i in func]

        x, y = self.pareto_optimal_front()

        plt.xlabel('Function 1', fontsize=15)
        plt.ylabel('Function 2', fontsize=15)
        plt.scatter(function1, function2)
        plt.plot(x, y, color='tab:orange')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.grid(ls='dashed', color='gray', alpha=0.3)
        plt.show()
