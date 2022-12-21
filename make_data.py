#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calc(x):
    A = np.array([[2.5, 0.5], [-0.5, 2.0]])
    b = np.array([10, 10])
    y = A @ x + b

    return y


def load_data():
    X = np.linspace(1, 10, 10)
    Y = np.linspace(1, 10, 10)

    xy = [(x, y) for x in X for y in Y]

    return xy


def plot(df:pd.DataFrame):
    ax = df.plot(x='input_x', y='input_y',
                 marker='o', alpha=0.7,
                 xlim=(0, 100), ylim=(0, 100),
                 label='original')
    df.plot(x='output_x', y='output_y',
            marker='o', alpha=0.7,
            xlim=(0, 100), ylim=(0, 100),
            label='projected',
            ax=ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.grid(ls='dashed', color='gray', alpha=0.3)
    ax.set_aspect('equal', 'box')

    plt.show()


def run():
    input_x = []
    input_y = []
    output_x = []
    output_y = []

    xy = load_data()
    for _xy in xy:
        (ox, oy) = calc(np.array(_xy))
        input_x.append(_xy[0])
        input_y.append(_xy[1])
        output_x.append(ox)
        output_y.append(oy)

    df = pd.DataFrame({"input_x":input_x,
                       "input_y":input_y,
                       "output_x":output_x,
                       "output_y":output_y})

    return df


    



if __name__ == '__main__':
    run()


