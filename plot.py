#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_sigmoid3D():
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = 1 / (1 + np.exp(-((X-5)**2 + (Y-5)**2))) +  1 / (1 + np.exp(-((X+5)**2 + (Y+5)**2)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

if __name__ == '__main__':
    plot_sigmoid3D()
