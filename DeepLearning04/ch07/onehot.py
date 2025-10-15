#!/usr/bin/env python

import numpy as np


def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]  # バッチの為の新しい軸を追加


state = (2, 0)
x = one_hot(state)
print(x.shape)  # (1, 12)
print(x)  # one-hotベクトルの表示
