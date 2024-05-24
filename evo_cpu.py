#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional


class EvoCPU:

    def __init__(self, filename, header=True) -> None:
        if not header:
            names = ["population", "chromosome", "time"]
            self.df = pd.read_csv(filename, names=names)
        else:
            self.df = pd.read_csv(filename)

    def make_average(self, skipnum=0) -> None:
        grouped = self.df.groupby(["population", "chromosome"])
        averaged = grouped.apply(lambda x: x.iloc[skipnum:].mean())
        self.heatmap_data = averaged["time"].unstack().T[::-1]

    def plot_combined(self,
                      pivoted:bool=True,
                      vmin:int=0, vmax:int=10000,
                      elev:int=20, azim:int=240) -> None:
        if pivoted:
            #dataheat = self.heatmap_data.iloc[::-1]
            dataheat = self.heatmap_data
            data3d = self.heatmap_data
        else:
            dataheat = self.df.pivot(columns="population",
                                     index="chromosome",
                                     values="time").iloc[::-1]
            dataheat = self.df.pivot(columns="population",
                                     index="chromosome",
                                     values="time")[::-1]

        fig = plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        ax = sns.heatmap(dataheat,
                         square=True,
                         vmin=vmin, vmax=vmax,
                         cmap="viridis", cbar=True,
                         xticklabels=True, yticklabels=True)
        cbar = ax.collections[0].colorbar
        cbar.set_label("Elapsed Time [ms]", fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        tick_labels = [0, 128, 256, 384, 512, 640, 768, 896, 1024]
        ax.set_xticks([i / 32 - 0.5 for i in tick_labels])
        ax.set_xticklabels(tick_labels, fontsize=18)
        #ax.set_yticks([len(data3d) - (i / 32) + 0.5 for i in tick_labels])
        ax.set_yticks([len(data3d) - (i / 32) + 0.5 for i in tick_labels])
        #ax.set_yticks([len(data3d) - (i / 32) + 0.5 for i in tick_labels[::-1]])
        ax.set_yticklabels(tick_labels, fontsize=18)
        #ax.set_yticklabels(tick_labels[::-1], fontsize=18)
        ax.set_xlabel("Population", fontsize=18)
        ax.set_ylabel("Chromosome", fontsize=18)
        ax = plt.subplot(1, 2, 2, projection="3d")
        x = np.arange(32, 1024+1, 32)
        #y = np.arange(32, 512+1, 32)
        y = np.arange(32, 1024+1, 32)[::-1]
        #y = np.arange(32, 1024+1, 32)
        X, Y = np.meshgrid(x, y)
        Z = data3d.values

        # surface plot
        surf = ax.plot_surface(X, Y, Z, 
                               vmin=vmin, vmax=vmax,
                               cmap="viridis")
        ax.view_init(elev=elev, azim=azim)

        # colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Elapsed Time [ms]", fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        # Set tick label size for X, Y, and Z axes
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="z", labelsize=14)
        ax.set_zlim(vmin, vmax)

        # Set axis labels
        ax.set_xlabel("Population", fontsize=18)
        ax.set_ylabel("Chromosome", fontsize=18)
        #ax.set_zlabel("Elapsed Time [ms]", fontsize=18)

        # Plot the figure
        plt.tight_layout()
        plt.show()
