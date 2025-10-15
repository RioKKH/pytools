#!/usr/bin/env python

import numpy as np
import dezero.layers as L

linear = L.Linear(10)  # 出力サイズだけを指定

batch_size, input_size = 100, 5
x = np.random.randn(batch_size, input_size)
y = linear(x)

print(f"y shape {y.shape}")
print(f"params shape: {linear.W.shape}, {linear.b.shape}")

for param in linear.params():
    print(param.name, param.shape)
