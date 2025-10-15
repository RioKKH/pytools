#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
import dezero.layers as L
import dezero.functions as F

# データセットの生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.1
iters = 10000


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(10, 1)

for i in range(iters):
    y_pred = model.forward(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

print("====")
print(f"W1 = {model.l1.W.data}")
print(f"b1 = {model.l1.b.data}")
print(f"W2 = {model.l2.W.data}")
print(f"b2 = {model.l2.b.data}")

# Plot
plt.scatter(x, y, s=10)
plt.xlabel("x")
plt.ylabel("y")
t = np.arange(0, 1, 0.01)[:, np.newaxis]
y_pred = model.forward(t)
plt.plot(t, y_pred.data, color="r")
plt.show()
