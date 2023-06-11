#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class Perceptron():

    def __init__(self):
        # training data
        self.X = np.array([[1, 1.2],  [1, 0.2],  [1, -0.2],
                           [1, -0.5], [1, -1.0], [1, -1.5]])
        # class label
        self.t = np.array([1, 1, 1, -1, -1, -1])
        # Initialize weight coefficients
        self.w = np.array([0.5, 0.5])
        #self.w = np.array([0.5, 0.5]).reshape([-1, 1])

        # Learning coefficients
        self.roh = 0.5 # 学習係数ρ

    def descriminant_function(self, x):
        return np.where(x >= 0, 1, -1)

    def evaluate(self, w, x, t):
        y = self.descriminant_function(np.sum(w * x))
        # 識別関数と正解が異なれば重みを更新
        print(f"%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d"
              %(x[0], x[1], w[0], w[1], np.sum(w * x), t, y))

        if (t != y):
            print("before:", t, w, x)
            print(self.roh * t * x)
            w += self.roh * t * x
            print("after:", t, w, x)
        return w

    def train(self):
        # training with simple perceptron
        num_of_loop = 1
        while 1:
            print("\n", num_of_loop, "巡目", "w=", self.w)
            for i in range(self.t.size):
                self.w = self.evaluate(self.w, self.X[i], self.t[i])

            y = self.descriminant_function(np.sum(self.w * self.X, 1))
            num_of_loop += 1

            if all(y == self.t):
                print("\n", num_of_loop, "巡目", "w=", self.w)
                for i in range(self.t.size):
                    w = self.evaluate(self.w, self.X[i], self.t[i])
                break

    def predict(self):
        pass

if __name__ == '__main__':
    perceptron = Perceptron()
    perceptron.train()

