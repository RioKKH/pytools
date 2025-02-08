#!/usr/bin/env python

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pysamoo.models.kriging import KrigingModel
from pysamoo.surrogate import Surrogate


# Step 1: 実際の評価関数を定義
class QuadraticProblem(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=1, nconstr=0, xl=-5, xu=5, type_var=np.float64)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.sum(X**2, axis=1)


# Step 2: サロゲートモデルの準備
class SurrogateQuadratic(Surrogate):
    def __init__(self):
        super().__init__()
        self.model = KrigingModel()  # サロゲートとしてKrigingを使用

    def fit(self, X, F):
        self.model.train(X, F)

    def predict(self, X):
        return self.model.predict(X)


# Step 3: メインのアルゴリズム処理
def main():
    # 実評価関数とサロゲートモデルを準備
    problem = QuadraticProblem()
    surrogate = SurrogateQuadratic()

    # 初期データを生成してサロゲートモデルを学習
    X_train = np.random.uniform(-5, 5, (10, 1))
    F_train = np.sum(X_train**2, axis=1).reshape(-1, 1)
    surrogate.fit(X_train, F_train)

    # サロゲートを使ったGAを実行する
    algorithm = GA(pop_size=20)  # 個体数20
    result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=("n_gen", 20),  # 最大20世代
        verbose=True,
        surrotage=surrogate,
    )

    # 結果を表示
    print("Best Solution (real evaluation): ", result.X)
    print("Best Objective (real evaluation): ", result.F)


if __name__ == "__main__":
    main()
