#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import traceback
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


class RollingQueue:

    def __init__(self, max_size=None):
        self.queue = deque([0.0] * max_size, maxlen=max_size)

    def append(self, value):
        self.queue.append(value)

    def show(self):
        for i in list(self.queue):
            print(f"{i},", end="")
        print()

    def len(self):
        return len(self.queue)

    def max(self):
        return np.max(list(self.queue))

    def min(self):
        return np.min(list(self.queue))

    def mean(self):
        return np.mean(list(self.queue))

    def variance(self):
        return np.var(list(self.queue))


class Perceptron:

    def __init__(self):
        # training data
        self.X = np.array([[1, 1.2],  [1, 0.2],  [1, -0.2],
                           [1, -0.5], [1, -1.0], [1, -1.5]])
        # class label
        self.t = np.array([1, 1, 1, -1, -1, -1])
        # Initialize weight coefficients
        self.wini = None
        self.w = None
        self.w0 = [] # for plot
        self.w1 = [] # for plot

        # Learning coefficients
        self.roh = 0.5 # 学習係数ρ
        self.beta = 0.01
        self.num_of_loop = 1
        self.epsilon = 1E-6
        self.max_size_of_queue = 4

        # Queue
        self.last4w0 = RollingQueue(max_size=self.max_size_of_queue)
        self.last4w1 = RollingQueue(max_size=self.max_size_of_queue)


    def init(self):
        self.__init__()


    def descriminant_function(self, x):
        return np.where(x >= 0, 1, -1)


    def evaluate(self, w, x, t, fw):
        y = self.descriminant_function(np.sum(w * x))
        # 識別関数と正解が異なれば重みを更新
        print(f"%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d"
              %(x[0], x[1], w[0], w[1], np.sum(w * x), t, y))
        fw.write(f"%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d\n"
                 %(x[0], x[1], w[0], w[1], np.sum(w * x), t, y))
        if (t != y):
            #dw = self.roh * t * x
            #if False:
            #if (self.last4w0.len() < self.max_size_of_queue):
            if (self.last4w0.len() < 2):
                dw = self.roh * t * x
            else:
                w0max = self.last4w0.max()
                w1max = self.last4w1.max()
                w0min = self.last4w0.min()
                w1min = self.last4w1.min()
                w0mean = self.last4w0.mean() 
                w1mean = self.last4w1.mean() 
                w0abs = np.abs(w0max - w0min)
                w1abs = np.abs(w1max - w1min)
                eps = self.epsilon

                roh0 = 1.5 * (np.abs(w0mean) + self.epsilon)
                roh1 = 1.5 * (np.abs(w1mean) + self.epsilon)

                #roh0 = (np.abs(w0mean) + eps) / (np.abs(w0max -w0min) + eps)
                #roh1 = (np.abs(w1mean) + eps) / (np.abs(w1max -w1min) + eps)

                #roh0 = (w0mean + eps) / (np.abs(w0max -w0min) + eps)
                #roh1 = (w1mean + eps) / (np.abs(w1max -w1min) + eps)

                dw0 = self.beta*(self.roh * t * x[0]) + (1-self.beta)*(roh0 * t * x[0])
                dw1 = self.beta*(self.roh * t * x[1]) + (1-self.beta)*(roh1 * t * x[1])
                dw = [dw0, dw1]

            w += dw
            print(f"\nold w0: {w[0]:.2f}, new w0: {w[0] + dw[0]:.2f}")
            print(f"old w1: {w[1]:.2f}, new w1: {w[1] + dw[1]:.2f}\n")
            self.last4w0.show()
            self.last4w1.show()
            self.last4w0.append(dw[0])
            self.last4w1.append(dw[1])

        self.w0.append(w[0])
        self.w1.append(w[1])
        return w


    def train(self):
        # training with simple perceptron
        self.num_of_loop = 1
        fw = open(f"weight_%.2f_%.2f.log" % (self.wini[0], self.wini[1]), "w")

        while 1:
            print("\n", self.num_of_loop, "巡目", "w=", self.w)
            fw.write("\n%d巡目 w=[%.2f, %.2f]\n"
                     % (self.num_of_loop, self.w[0], self.w[1]))

            for i in range(self.t.size):
                self.w = self.evaluate(self.w, self.X[i], self.t[i], fw)

            y = self.descriminant_function(np.sum(self.w * self.X, 1))
            self.num_of_loop += 1

            if all(y == self.t):
                print("\n", self.num_of_loop, "巡目", "w=", self.w)
                fw.write("\n%d巡目 w=[%.2f, %.2f]\n"
                         % (self.num_of_loop, self.w[0], self.w[1]))

                for i in range(self.t.size):
                    w = self.evaluate(self.w, self.X[i], self.t[i], fw)
                break

            if self.num_of_loop >= 20:
                raise ValueError

        fw.close()


    def set_weight(self, w0, w1):
        self.wini = np.array([w0, w1])
        self.w = np.array([w0, w1])
        self.w0.append(w0)
        self.w1.append(w1)
        self.last4w0.append(w0)
        self.last4w1.append(w1)


    def save(self, fout='result.csv'):
        with open(fout, 'a') as fw:
            fw.write("%.2f,%.2f,%d\n" 
                     % (self.wini[0], self.wini[1], self.num_of_loop))


    def plot(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        ax.plot(self.w0, self.w1,
                marker='o', ms=2, color='black', alpha=0.5, ls='-')
        ax.scatter(self.w0[0], self.w1[0], s=40, color='red')
        ax.scatter(self.w0[-1], self.w1[-1], s=40, color='green')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(ls='dashed', color='gray', alpha=0.5)
        plt.title(f"Initial weight (w0, w1) = (%.2f, %.2f)"
                  % (self.wini[0], self.wini[1]))
        plt.savefig(f"weight_%.2f_%.2f.png" % (self.wini[0], self.wini[1]))
        plt.cla()
        plt.clf()


def parse_double(value):
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid double value: {value}")


if __name__ == '__main__':
    eps = 1E-6
    #X = [-1.0]
    #Y = [1.75]
    #X = [ -2.0, -1.0, -0.5, -0.25]
    #Y = [ -2.0,  0.0,  1.75]
    X = np.arange(-2, 2+eps, 0.25)
    Y = np.arange(-2, 2+eps, 0.25)
    parser = argparse.ArgumentParser(
        description = "Simple perceptron"
    )
    parser.add_argument(
        '-w', '--weight',
        dest='w', nargs=2, type=parse_double,
        help='initial weight as list'
    )
    args = parser.parse_args()

    try:
        perceptron = Perceptron()
        for x in X:
            for y in Y:
                perceptron.init()
                perceptron.set_weight(x, y)
                perceptron.train()
                #perceptron.plot()
                perceptron.save()

    except Exception as e:
        perceptron.plot()
        print(e)
        print(traceback.format_exc())

