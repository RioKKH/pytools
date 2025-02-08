#!/usr/bin/env python
# coding : utf-8

from opthub_client.api import OptHub

with OptHub("owd9miCKJ39thx84aEBj71jL79HQvaRB6DU5h6Of") as api:
    opthub_match = api.match("30e7deb3-f11d-4343-b7ce-055996447bbd")
    trial = opthub_match.submit([1.23, 4.56, 7.89])

    # 評価がおわるまで待機して評価値を取得する
    eval = trial.wait_evaluation()

    # 評価に格納されている目的関数の値 (scalarの場合) を取得して出力
    # 多目的の場合は目的関数の値がベクトルなので、objective.vectorで参照
    print("Evaluation completed. Objective function value: " + str(eval.objective.scalar))

    # スコア計算がおわるまで待機してスコアを取得
    score = trial.wait_scoring()

    # スコアの値を取得
    print("Scoring completed. score: " + str(score.value))
