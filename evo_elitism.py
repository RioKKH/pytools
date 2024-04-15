#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EvoElitism:

    def __init__(self, filename):
        self.filename = filename
        self.load()

    def load(self):
        names = ['population', 'chromosome', 'timepercent', 'time', 'count']
        self.df = pd.read_csv(self.filename, names=names)

    def plot_3d(self, vmin=0, vmax=30):
        # ピボットテーブルを生成する
        data = self.df.pivot_table(columns="population",
                                   index="chromosome",
                                   values="time")

        # X軸、Y軸、Z軸の値を取得・生成する
        x = np.arange(32, 1024+1, 32)
        y = np.arange(32, 1024+1, 32)
        X, Y = np.meshgrid(x, y)
        Z = data.values / 1E6

        # 3Dプロットを初期化する
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z,
                               cmap='jet',
                               vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel("Population")
        ax.set_ylabel("Chromosome")
        ax.set_zlabel("Time [ms]")

        plt.show()

