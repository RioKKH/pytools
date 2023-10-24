#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def func1(x):
    return np.sin(x/2 - 2)

def func2(x):
    return np.sin(x/2 - 2) + np.cos(10*x)/6

def func3(x):
    return np.sin(x) + np.cos(7 * x)/4 + x/100

def func4(x):
    return np.sin(x*10) + np.sin(x * 7 + 1)/10


def main():
    x = np.linspace(-10, 10, 1000)

    y1 = func1(x)
    y2 = func2(x)
    y3 = func3(x)
    y4 = func4(x)

    plt.plot(x, y1, label='sin(x/2 - 2)')
    plt.grid(ls='dashed', color='gray', alpha=0.5)
    plt.xlim(-5, 7)
    plt.legend()
    plt.show()

    plt.plot(x, y2, label='sin(x/2 - 2) + cos(10*x)/6')
    plt.grid(ls='dashed', color='gray', alpha=0.5)
    plt.xlim(-5, 7)
    plt.legend()
    plt.show()

    plt.plot(x, y3, label='sin(x) + cos(7 * x)/4 + x/100')
    plt.grid(ls='dashed', color='gray', alpha=0.5)
    plt.xlim(-5, 7)
    plt.legend()
    plt.show()

    plt.plot(x, y4, label='sin(x*10) + sin(x * 7 + 1)/10')
    plt.grid(ls='dashed', color='gray', alpha=0.5)
    plt.xlim(-5, 7)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

