#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple

class NSGA2:
    """ NSGA-IIアルゴリズムの実装クラス

    このクラスは、 Non-dominated Sorting Genetic Algorithm II（NSGA-II）を実装
    したものです。NSGA-IIは、多目的最適化問題を解くための進化計算アルゴリズムです。

    Attributes:
        pop_size (int): 個体群のサイズ
        n_obj (int): 目的関数の数
        n_var (int): 決定変数の数
        xl (np.ndarray): 決定変数の下限値
        xu (np.ndarray): 決定変数の上限値
        problem: 最適化問題を定義するオブジェクト。評価関数 `evaluate`を持つ
        必要があります。
    """

    def __init__(self, pop_size: int, n_obj: int, n_var: int, 
                 xl: np.ndarray, xu: np.ndarray, problem) -> None:
        """
        Args:
            pop_size (int): 個体群のサイズ
            n_obj (int): 目的関数の数
            n_var (int): 決定変数の数
            xl (np.ndarray): 決定変数の下限値
            xu (np.ndarray): 決定変数の上限値
            problem: 最適化問題を定義するオブジェクト。評価関数 `evaluate`を持つ
        """
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.xl = xl
        self.xu = xu
        self.problem = problem

    def run(self, n_gen: int) -> Tuple[np.ndarray, np.ndarray]:
        """NSGA-IIアルゴリズムを実行するメソッド

        Args:
            n_gen (int): 世代数

        Returns:
            Tuple[np.ndarray, np.ndarray]: 最終的な解集団(X)と目的関数値(F)。
        """
        # 初期個体群の生成
        X = self._initialize_population()
        F = self.problem.evaluate(X)

        for _ in range(n_gen):
            # 子個体の生成
            offspring_X = self._create_offspring(X)
            offspring_F = self.problem.evaluate(offspring_X)

            # 親個体と子個体を結合
            X = np.vstack([X, offspring_X])
            F = np.vstack([F, offspring_F])

            # 非裂開のソーティング
            fronts = self._non_dominated_sort(F)

            # 次世代の個体を選択
            X, F = self._select_next_population(X, F, fronts)

        return X, F

    def _initialize_population(self) -> np.ndarray:
        """初期個体群を生成するメソッド

        Returns:
            np.ndarray: 初期個体群
        """
        return np.random.uniform(self.xl, self.xu, size=(self.pop_size, self.n_var))

    def _create_offspring(self, X: np.ndarray) -> np.ndarray:
        """子個体を生成するメソッド

        Args:
            X (np.ndarray): 親個体群

        Returns:
            np.ndarray: 子個体群
        """
        # ここでは単純な一様交叉と突然変異を実装
        offspring = np.empty((self.pop_size, self.n_var))
        for i in range(0, self.pop_size, 2):
            p1, p2 = np.random.choice(self.pop_size, 2, replace=False)
            # 一様交叉
            mask = np.random.rand(self.n_var) > 0.5
            offspring[i] = np.where(mask, X[p1], X[p2])
            offspring[i + 1] = np.where(mask, X[p2], X[p1])

        

