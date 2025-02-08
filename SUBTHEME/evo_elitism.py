#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


class EvoElitism:

    def __init__(self, filename, header=True):
        self.filename = filename
        self.load(header=header)


    def load(self, header=False):
        if not header:
            names = ['population', 'chromosome', 'timepercent', 'time', 'count']
            self.df = pd.read_csv(self.filename, names=names)
            # time列の値をミリ秒に変換する
            self.df["time"] = self.df["time"] / 1E6
        else:
            self.df = pd.read_csv(self.filename)


    def make_average(self,
                     skipnum: int=5,
                     tgt_function: str="elitism") -> pd.DataFrame:

        # kernel_functions = ["evaluation", "selection", "crossover",
        #                     "mutation", "elitism", "replacewithelites"]

        grouped = self.df.groupby(["population", "chromosome"])
        # 以下のコードが何をしているかを理解したければ、以下のコメントを
        # 外して実行してみると良い
        # for name, group in grouped:
        #     print(f"Group: {name}")
        #     print(group.head(5)) # スキップしたい最初の5行を表示する
        averaged = grouped.apply(
            lambda x: x.sort_values(by=tgt_function).iloc[2:-3].mean()
            if len(x) > 5 else x.mean()
        )
        #averaged = grouped.apply(lambda x: x.iloc[skipnum:-1].mean())
        # groupbyオブジェクトをunstackメソッドでピボットテーブルに変換する
        # その後、T属性で転置して、行と列を反転させる。これによって
        # X軸がpopulation、Y軸がchromosomeとなるようにする
        #self.heatmap_data = averaged[tgt_function].unstack().T
        self.heatmap_data = averaged[tgt_function].unstack().T[::-1]

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
                               cmap='plasma',
                               #cmap='viridis',
                               #cmap='jet',
                               vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel("Population")
        ax.set_ylabel("Chromosome")
        ax.set_zlabel("Time [ms]")

        plt.show()

    def plot_combined(self, 
                      pivoted=True, vmin=0, vmax=30, elev=20, azim=240):
        if pivoted:
            dataheat = self.heatmap_data
            #dataheat = self.heatmap_data.iloc[::-1]
            data3d = self.heatmap_data
        else:
            dataheat = self.df.pivot_table(columns="population",
                                           index="chromosome",
                                           values='time')[::-1]
            data3d = self.df.pivot_table(columns="population",
                                         index="chromosome",
                                         values='time')[::-1]
        print(
            f"min={dataheat.values.min():.4f}, "
            f"max={dataheat.values.max():.4f}, "
            f"mean={dataheat.values.mean():.4f}"
        )

        fig = plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        ax = sns.heatmap(dataheat,
                         square=True,
                         vmin=vmin, vmax=vmax,
                         #cmap='plasma',
                         cmap='viridis',
                         #cmap='jet',
                         cbar=True,
                         xticklabels=True, yticklabels=True)
        cbar = ax.collections[0].colorbar
        cbar.set_label("Elapsed Time [ms]", fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        tick_labels = [0, 128, 256, 384, 512, 640, 768, 896, 1024]
        ax.set_xticks([i / 32 -0.5 for i in tick_labels])
        ax.set_xticklabels(tick_labels, fontsize=18)
        ax.set_yticks([len(data3d) - (i / 32) + 0.5 for i in tick_labels[::-1]])
        ax.set_yticklabels(tick_labels[::-1], fontsize=18)

        ax.set_xlabel("Population", fontsize=18)
        ax.set_ylabel("Chromosome", fontsize=18)

        ax = plt.subplot(1, 2, 2, projection='3d')
        # X軸、Y軸、Z軸の値を取得・生成する
        x = np.arange(32, 1024+1, 32)
        y = np.arange(32, 1024+1, 32)[::-1]
        X, Y = np.meshgrid(x, y)
        Z = data3d.values

        # surfaceプロットを作成する
        surf = ax.plot_surface(X, Y, Z, cmap="viridis",
        #surf = ax.plot_surface(X, Y, Z, cmap="plasma",
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
        ax.set_zlim(vmin, vmax)

        # 軸ラベルを設定する
        ax.set_xlabel("Population", fontsize=18)
        ax.set_ylabel("Chromosome", fontsize=18)
        ax.set_zlabel("Elapsed Time [ms]", fontsize=18)

        plt.tight_layout()
        plt.show()


    
def plot_comparison(lhs, rhs, kind="ratio", vmin=0, vmax=30, elev=20, azim=240):

    if kind == "ratio":
        comp = lhs / rhs
    elif kind == "diff":
        comp = lhs - rhs

    print(comp)
    print(
        f"min={comp.values.min():.4f}, "
        f"max={comp.values.max():.4f}, "
        f"mean={comp.values.mean():.4f}"
    )
    dataheat = comp
    data3d = comp

    fig = plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    ax = sns.heatmap(dataheat,
                     square=True,
                     vmin=vmin, vmax=vmax,
                     cmap='viridis',
                     #cmap='jet',
                     cbar=True,
                     xticklabels=True, yticklabels=True)
    cbar = ax.collections[0].colorbar
    if kind == "ratio":
        cbar.set_label("Execution Time Ratio", fontsize=18)
    elif kind == "diff":
        cbar.set_label("Execution Time Difference [msec]", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    tick_labels = [0, 128, 256, 384, 512, 640, 768, 896, 1024]
    ax.set_xticks([i / 32 -0.5 for i in tick_labels])
    ax.set_xticklabels(tick_labels, fontsize=18)
    ax.set_yticks([len(data3d) - (i / 32) + 0.5 for i in tick_labels])
    #ax.set_yticks([len(data3d) - (i / 32) + 0.5 for i in tick_labels[::-1]])
    ax.set_yticklabels(tick_labels, fontsize=18)

    ax.set_xlabel("Population", fontsize=18)
    ax.set_ylabel("Chromosome", fontsize=18)

    ax = plt.subplot(1, 2, 2, projection='3d')
    # X軸、Y軸、Z軸の値を取得・生成する
    x = np.arange(32, 1024+1, 32)
    #y = np.arange(32, 1024+1, 32)
    y = np.arange(32, 1024+1, 32)[::-1]
    X, Y = np.meshgrid(x, y)
    Z = data3d.values

    # surfaceプロットを作成する
    #surf = ax.plot_surface(X, Y, Z, cmap="jet",
    surf = ax.plot_surface(X, Y, Z, cmap="viridis",
                           vmin=vmin, vmax=vmax)
    ax.view_init(elev=elev, azim=azim)
    
    # カラーバーを作成する
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    if kind == "ratio":
        cbar.set_label("Execution Time Ratio", fontsize=18)
    elif kind == "diff":
        cbar.set_label("Execution Time Difference [msec]", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Set tick label size for X, Y, and Z axes
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    ax.set_zlim(vmin, vmax)

    # 軸ラベルを設定する
    ax.set_xlabel("Population", fontsize=18)
    ax.set_ylabel("Chromosome", fontsize=18)

    if kind == "ratio":
        ax.set_zlabel("Execution Time Ratio", fontsize=18)
    elif kind == "diff":
        ax.set_zlabel("Execution Time Difference [msec]", fontsize=18)


    plt.tight_layout()
    plt.show()


    
