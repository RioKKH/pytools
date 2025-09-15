#!/usr/bin/env python

import numpy as np

"""
1台のスロットを10海プレイして行動価値の推定値Q_nを求める
"""

np.random.seed(0)  # シードを固定する
rewards = []
Q = 0

for n in range(1, 11):  # 1から10まで繰り返す
    reward = np.random.rand()  # ダミーの報酬

    # 非効率的な書き方　時間・空間計算量が大きくなる
    # rewards.append(reward)
    # Q = sum(rewards) / n  # 報酬の推定値

    # インクリメンタル(逐次的な)実装
    # Q = Q + (reward - Q) / n
    Q += (reward - Q) / n
    print(Q)
