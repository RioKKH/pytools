#!/usr/bin/env python

import numpy as np

"""
1台のスロットを10海プレイして行動価値の推定値Q_nを求める
"""

np.random.seed(0)  # シードを固定する
rewards = []

for n in range(1, 11):  # 1から10まで繰り返す
    reward = np.random.rand()  # ダミーの報酬
    rewards.append(reward)
    Q = sum(rewards) / n  # 報酬の推定値
    print(Q)
