#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(fin:str) -> pd.DataFrame:
    names = ('POPULATION', 'CHROMOSOME', 'ELAPSEDTIME', 'FITMAX', 'FITMIN', 'FITMEAN')
    df = pd.read_csv(fin, names=names)
    return df


def load_data_CPU(fin:str) -> pd.DataFrame:
    names = ("gen", "FITMEAN", "FITMIN", "FITMAX", "FITSTD")
    df = pd.read_csv(fin, names=names)
    return df


def load_stats(fin:str) -> pd.DataFrame:
    names = ('gen', 'fitmean', 'fitmin', 'fitmax', 'fitstd')
    df = pd.read_csv(fin,
                     index_col=0,
                     names = names)
    return df


def load_trend_data(fin:str) -> pd.DataFrame:
    names = ('gen', 'pop', 'chrom', 'elapsedtime', 'fitmax', 'fitmin', 'fitmean')
    df = pd.read_csv(fin, names=names)

    return df


def load_elapsed_time(fin:str) -> pd.DataFrame:
    names = ("POPULATION", "CHROMOSOME", "ELAPSEDTIME")
    df = pd.read_csv(fin,
                     names = names,)

    df.sort_values(by=["POPULATION", "CHROMOSOME"],
                   ascending=[True, True],
                   inplace=True)
    return df


def make_average(df:pd.DataFrame) -> pd.DataFrame:
    df_averaged = df.groupby(["POPULATION", "CHROMOSOME"])["ELAPSEDTIME"].mean()
    df_averaged = df_averaged.reset_index()
    df_averaged = df_averaged.sort_values(["POPULATION", "CHROMOSOME"])
    return df_averaged

def compare_stats(dfCPU: pd.DataFrame,
                  dfGPU: pd.DataFrame) -> None:
    ax = dfCPU.fitmean.plot()
    dfGPU.fitmean.plot(ax=ax)
    ax.axhline(y=1024, alpha=0.5, color='black')

    ax.grid(ls='dashed', color='gray', alpha=0.5)
    ax.legend(['CPU', 'GPU'])
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    ax.set_ylim(500, 1024)
    plt.show()

def plot(df, vmin=0, vmax=10000, title=None):
    data = df.pivot_table(columns = "POPULATION",
                          index   = "CHROMOSOME",
                          values  = "ELAPSEDTIME")[::-1]

    ax = sns.heatmap(data,
                     square=True,
                     vmin=vmin, vmax=vmax,
                     cbar=True,
                     cmap='jet',
                     xticklabels=True, yticklabels=True)

    # Customize tick labels
    tick_labels = [32, 128, 256, 384, 512, 640, 768, 896, 1024]
    ax.set_xticks([i / 32 - 0.5 for i in tick_labels])
    ax.set_xticklabels(tick_labels, fontsize=14)
    ax.set_yticks([len(data) - (i / 32) + 0.5 for i in tick_labels[::-1]])
    ax.set_yticklabels(tick_labels[::-1], fontsize=14)

    # Set colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Elapsed time [ms]', fontsize=18)

    # Adjust colorbar tick label size
    cbar.ax.tick_params(labelsize=14)

    # Set axis labels fontsize
    ax.set_xlabel('Population', fontsize=18)
    ax.set_ylabel('Chromosome', fontsize=18)

    # Set the title if specified
    if title:
        plt.title(title, fontsize=18)

    plt.tight_layout()
    plt.show()

def plot_3d(df, vmin=0, vmax=10000):
    # ピボットデーブルを作成する
    data = df.pivot_table(columns = "POPULATION",
                          index="CHROMOSOME",
                          values="ELAPSEDTIME")
                          #values="ELAPSEDTIME")[::-1]

    # X軸、Y軸、Z軸の値を取得する
    x = np.arange(32, 1024+1, 32)
    y = np.arange(32, 1024+1, 32)
    X, Y = np.meshgrid(x, y)
    Z = data.values

    # 3Dプロットを初期化する
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    # surfaceプロットを作成する
    surf = ax.plot_surface(X, Y, Z,
                           cmap = 'jet',
                           vmin = vmin, vmax = vmax)
    # カラーバーを作成する
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 軸ラベルを設定する
    ax.set_xlabel("Population")
    ax.set_ylabel("Chromosome")
    ax.set_zlabel("Elapsed time [ms]")

    plt.show()

def plot_combined(df,
                  vmin=0, vmax=10000,
                  elev=20, azim=240):
    dataheat = df.pivot_table(columns = "POPULATION",
                              index   = "CHROMOSOME",
                              values  = "ELAPSEDTIME")[::-1]

    data3d = df.pivot_table(columns = "POPULATION",
                            index="CHROMOSOME",
                            values="ELAPSEDTIME")[::-1]

    fig = plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    ax = sns.heatmap(dataheat,
                     square=True,
                     vmin=vmin, vmax=vmax,
                     cbar=True,
                     cmap='jet',
                     xticklabels=True, yticklabels=True)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Elapsed time [ms]', fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    tick_labels = [32, 128, 256, 384, 512, 640, 768, 896, 1024]
    ax.set_xticks([i / 32 - 0.5 for i in tick_labels])
    ax.set_xticklabels(tick_labels, fontsize=18)
    ax.set_yticks([len(data3d) - (i / 32) + 0.5 for i in tick_labels[::-1]])
    ax.set_yticklabels(tick_labels[::-1], fontsize=18)

    ax.set_xlabel('Population', fontsize=18)
    ax.set_ylabel('Chromosome', fontsize=18)

    ax = plt.subplot(1, 2, 2, projection='3d')
    # X軸、Y軸、Z軸の値を取得する
    x = np.arange(32, 1024+1, 32)
    #y = np.arange(1024, 32-1, 32)
    y = np.arange(32, 1024+1, 32)[::-1]
    X, Y = np.meshgrid(x, y)
    Z = data3d.values

    # surfaceプロットを作成する
    surf = ax.plot_surface(X, Y, Z,
                           cmap = 'jet',
                           vmin = vmin, vmax = vmax)
    ax.view_init(elev=elev, azim=azim)

    # カラーバーを作成する
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Elapsed time [ms]', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    #cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Set tick label size for X, Y, Z axes
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    # 軸ラベルを設定する
    ax.set_xlabel("Population", fontsize=18)
    ax.set_ylabel("Chromosome", fontsize=18)
    ax.set_zlabel("Elapsed time [ms]", fontsize=18)


    plt.tight_layout()
    plt.show()

def plot_combined_ratio(dfcpu, dfgpu,
                        vmin=0, vmax=50,
                        elev=20, azim=240):

    def _load(df):
        data = df.pivot_table(columns = "POPULATION",
                              index   = "CHROMOSOME",
                              values  = "ELAPSEDTIME")[::-1]
        return data

    cpu = _load(dfcpu)
    gpu = _load(dfgpu)
    diff = cpu - gpu
    ratio = cpu / gpu

    fig = plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    ax = sns.heatmap(diff,
    #ax = sns.heatmap(ratio,
                     square=True,
                     vmin=vmin, vmax=vmax,
                     cbar=True,
                     cmap='jet',
                     xticklabels=True, yticklabels=True)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Difference [msec]', fontsize=18)
    #cbar.set_label('Ratio', fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    tick_labels = [32, 128, 256, 384, 512, 640, 768, 896, 1024]
    ax.set_xticks([i / 32 - 0.5 for i in tick_labels])
    ax.set_xticklabels(tick_labels, fontsize=18)
    ax.set_yticks([len(diff) - (i / 32) + 0.5 for i in tick_labels[::-1]])
    #ax.set_yticks([len(ratio) - (i / 32) + 0.5 for i in tick_labels[::-1]])
    ax.set_yticklabels(tick_labels[::-1], fontsize=18)

    ax.set_xlabel('Population', fontsize=18)
    ax.set_ylabel('Chromosome', fontsize=18)

    ax = plt.subplot(1, 2, 2, projection='3d')
    # X軸、Y軸、Z軸の値を取得する
    x = np.arange(32, 1024+1, 32)
    #y = np.arange(1024, 32-1, 32)
    y = np.arange(32, 1024+1, 32)[::-1]
    X, Y = np.meshgrid(x, y)
    Z = diff.values
    #Z = ratio.values

    # surfaceプロットを作成する
    surf = ax.plot_surface(X, Y, Z,
                           cmap = 'jet',
                           vmin = vmin, vmax = vmax)
    ax.view_init(elev=elev, azim=azim)
    # カラーバーを作成する
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Difference [msec]', fontsize=18)
    #cbar.set_label('Ratio', fontsize=18)
    #cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.ax.tick_params(labelsize=14)

    # Set tick label size for X, Y, Z axes
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    # 軸ラベルを設定する
    ax.set_xlabel("Population", fontsize=18)
    ax.set_ylabel("Chromosome", fontsize=18)
    ax.set_zlabel("Difference [msec]", fontsize=18)
    #ax.set_zlabel("Ratio", fontsize=18)


    plt.tight_layout()
    plt.show()


def make_heatmap(dfCPU, dfGPU, vmin=0, vmax=2000, rvmin=0, rvmax=30) -> None:
    def _plot(cgpu,
              vmin=vmin, vmax=vmax,
              bratio:bool = False):
        ax = sns.heatmap(cgpu, 
                         #annot=True, 
                         #fmt='.1f',
                         square=True,
                         vmin=vmin, vmax=vmax,
                         cbar=True,
                         cmap='jet',
                         xticklabels=True,
                         yticklabels=True)
        cbar = ax.collections[0].colorbar
        if bratio:
            cbar.set_label('Ratio')
        else:
            cbar.set_label('Elapsed time [ms]')
        plt.tight_layout()
        plt.show()

    cpu = dfCPU.pivot_table(columns = 'POPULATION',
                            index   = 'CHROMOSOME',
                            values  = 'ELAPSEDTIME')[::-1]
    gpu = dfGPU.pivot_table(columns = 'POPULATION',
                            index   = 'CHROMOSOME',
                            values  = 'ELAPSEDTIME')[::-1]
    ratio = cpu / gpu

    _plot(cpu, bratio=False)
    _plot(gpu, bratio=False)
    _plot(ratio, vmin=rvmin, vmax=rvmax, bratio=True)


