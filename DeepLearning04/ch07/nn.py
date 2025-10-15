#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

# データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 1. 重みの初期化
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))


# 2. ニューラルネットワークの推論
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000

# 3. ニューラルネットワークの学習
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()

    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:  # 1000回ごとに損失を表示
        print(loss.data)

print("====")
print("W1 =", W1.data)
print("b1 =", b1.data)
print("W2 =", W2.data)
print("b2 =", b2.data)

# Plot
# NumPy配列をそのまま使う場合: .data不要
# DeZeroのVariableオブジェクトを使う場合: .dataでNumPy配列を取り出すこと
plt.scatter(x, y, s=10)
plt.xlabel("x")
plt.ylabel("y")
# (100,): 1次元配列 --> (100, 1): 2次元配列(列ベクトル)
t = np.arange(0, 1, 0.01)[:, np.newaxis]
y_pred = predict(t)
# predict()が返すのはDeZeroのVariableオブジェクトの為、.dataでNumPy配列を取り出す事
plt.plot(t, y_pred.data, color="r")
plt.show()
