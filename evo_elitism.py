#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EvoElitism:

    def __init__(self, filename):
        self.filename = filename
        self.load()

    def load(self):
        names = ['population', 'chromosome', 'timepercent', 'time', 'count']
        self.df = pd.read_csv(self.filename, names=names)
        # time列の値をミリ秒に変換する
        self.df["time"] = self.df["time"] / 1E6

    def plot_3d(self, vmin=0, vmax=30):
        # ピボットテーブルを生成する
        data = self.df.pivot_table(columns="population",
                                   index="chromosome",
                                   values="time")

        # X軸、Y軸、Z軸の値を取得・生成する
        x = np.arange(32, 1024+1, 32)
        y = np.arange(32, 1024+1, 32)
        X, Y = np.meshgrid(x, y)
        Z = data.values

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

    def plot_combined(self, vmin=0, vmax=30, elev=30, azim=30):
        dataheat = self.df.pivot_table(columns="population",
                                       index="chromosome",
                                       values='time')[::-1]
        data3d = self.df.pivot_table(columns="population",
                                     index="chromosome",
                                     values='time')[::-1]
        fig = plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        ax = sns.heatmap(dataheat,
                         square=True,
                         vmin=vmin, vmax=vmax,
                         cmap='jet',
                         cbar=True,
                         xticklabels=True, yticklabels=True)
        cbar = ax.collections[0].colorbar
        cbar.set_label("Elapsed Time [ms]", fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        tick_labels = [0, 128, 256, 384, 512, 640, 768, 896, 1024]
        ax.set_xticks([i / 32 -0.5 for i in tick_labels])
        ax.set_xticklabels(tick_labels, fontsize=18)
        ax.set_yticks([len(data3d) - (i / 32) + 0.5 for i in tick_labels[::-1]])
        ax.set_yticklabels(tick_labels, fontsize=18)

        ax.set_xlabel("Population", fontsize=18)
        ax.set_ylabel("Chromosome", fontsize=18)

        ax = plt.subplot(1, 2, 2, projection='3d')
        # X軸、Y軸、Z軸の値を取得・生成する
        x = np.arange(32, 1024+1, 32)
        y = np.arange(32, 1024+1, 32)[::-1]
        X, Y = np.meshgrid(x, y)
        Z = data3d.values

        # surfaceプロットを作成する
        surf = ax.plot_surface(X, Y, Z, cmap="jet",
                               vmin=vmin, vmax=vmax)
        ax.view_init(elev=elev, azim=azim)
        
        # カラーバーを作成する
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Elapsed Time [ms]", fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        # Set tick label size for X, Y, and Z axes
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='z', labelsize=14)

        # 軸ラベルを設定する
        ax.set_xlabel("Population", fontsize=18)
        ax.set_ylabel("Chromosome", fontsize=18)
        ax.set_zlabel("Elapsed Time [ms]", fontsize=18)

        plt.tight_layout()
        plt.show()


