#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional


class Comparison:

    def __init__(self, 
                 cpu_heatmapdata,
                 gpu_heatmapdata,
                 flip_cpu: bool = True, kind="ratio") -> None:
        if flip_cpu:
            self.cpu = cpu_heatmapdata
        else:
            self.cpu = cpu_heatmapdata

        self.gpu = gpu_heatmapdata

        self.kind = kind
        if kind == "ratio":
            self.comp = self.cpu / self.gpu
        elif kind == "diff":
            self.comp = self.cpu - self.gpu
        else:
            raise ValueError("kind must be either 'ratio' or 'diff'.")

    def plot_combined(self,
                      vmin:int=0, vmax:int=1000,
                      elev:int=20, azim:int=240) -> None:

        dataheat = self.comp
        data3d = dataheat

        fig = plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        ax = sns.heatmap(dataheat,
                         square=True,
                         vmin=vmin, vmax=vmax,
                         cmap="viridis", cbar=True,
                         xticklabels=True, yticklabels=True)
        cbar = ax.collections[0].colorbar
        if self.kind == "ratio":
            cbar.set_label("Execution Time Ratio", fontsize=18)
        elif self.kind == "diff":
            cbar.set_label("Exection Time Difference [msec]", fontsize=18)
        #cbar.set_label("Elapsed Time [ms]", fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        tick_labels = [0, 128, 256, 384, 512, 640, 768, 896, 1024]
        ax.set_xticks([i / 32 - 0.5 for i in tick_labels])
        ax.set_xticklabels(tick_labels, fontsize=18)
        ax.set_yticks([len(data3d) - (i / 32) + 0.5 for i in tick_labels])
        ax.set_yticklabels(tick_labels, fontsize=18)

        ax.set_xlabel("Population", fontsize=18)
        ax.set_ylabel("Chromosome", fontsize=18)

        ax = plt.subplot(1, 2, 2, projection="3d")
        x = np.arange(32, 1024+1, 32)
        y = np.arange(32, 1024+1, 32)[::-1]
        X, Y = np.meshgrid(x, y)
        Z = data3d.values

        # surface plot
        surf = ax.plot_surface(X, Y, Z, 
                               vmin=vmin, vmax=vmax,
                               cmap="viridis")
        ax.view_init(elev=elev, azim=azim)

        # colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        if self.kind == "ratio":
            cbar.set_label("Execution Time Ratio", fontsize=18)
        elif self.kind == "diff":
                cbar.set_label("Exection Time Difference [msec]", fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        # Set tick label size for X, Y, and Z axes
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.tick_params(axis="z", labelsize=14)
        ax.set_zlim(vmin, vmax)

        # Set axis labels
        ax.set_xlabel("Population", fontsize=18)
        ax.set_ylabel("Chromosome", fontsize=18)
        #if self.kind == "ratio":
        #    ax.set_zlabel("Execution Time Ratio", fontsize=18)
        #elif self.kind == "diff":
        #    ax.set_zlabel("Exection Time Difference [msec]", fontsize=18)
        #ax.set_zlabel("Elapsed Time [ms]", fontsize=18)

        print(f"min={dataheat.values.min():.4f}, max={dataheat.values.max():.4f}")

        # Plot the figure
        plt.tight_layout()
        plt.show()
