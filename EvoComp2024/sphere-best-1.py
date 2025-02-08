#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cma
import numpy as np

from opthub_client.api import OptHub

MY_API_KEY = "owd9miCKJ39thx84aEBj71jL79HQvaRB6DU5h6Of"

# tutorial/sphere-best-1
MATCH_UUID = "ea854d4c-63fd-43ac-abf3-6790339d0e67"

with OptHub(MY_API_KEY) as api:
    opthub_match = api.match(MATCH_UUID)

    def objective(x: list | np.ndarray) -> float:
        """Calculate the objective function value."""
        if isinstance(x, np.ndarray):
            x = x.tolist()
        trial = opthub_match.submit(x)

        # 評価が終わるまで待機して評価額を取得
        eval = trial.wait_evaluation()

        # 評価に格納されている目的関数の値 (scalarの場合) を取得して出力
        # 多目的の場合は目的関数の値がベクトルなので、objective.vectorで参照
        print("Evaluation completed. Objective function value: " + str(eval.objective.scalar))

        return eval.objective.scalar


x0 = [1, 1]  # CMA-ESの初期平均ベクトル
sigma = 0.5  # CMA-ESの初期ステップサイズ
max_iter = 3 # 打ち切り世代数
es = cma.CMAEvolutionStrategy(x0, sigma, {"maxiter": max_iter})
es.optimize(objective)
