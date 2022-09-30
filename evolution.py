#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_stats(fin:str) -> pd.DataFrame:
    names = ('gen', 'fitmean', 'fitmin', 'fitmax', 'fitstd')
    df = pd.read_csv(fin,
                     index_col=0,
                     names = names)

    return df


def load_elapsed_time(fin:str) -> pd.DataFrame:
    names = ("POPULATION", "CHROMOSOME", "ELAPSEDTIME")
    df = pd.read_csv(fin,
                     names = names,)

    df.sort_values(by=["POPULATION", "CHROMOSOME"],
                   ascending=[True, True],
                   inplace=True)
    return df


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

def plot(df, vmin=0, vmax=10000):
    gpu = df.pivot_table(columns = "POPULATION",
                         index   = "CHROMOSOME",
                         values  = "ELAPSEDTIME")[::-1]

    sns.heatmap(gpu,
                square=True,
                vmin=vmin, vmax=vmax,
                cbar=True,
                xticklabels=True, yticklabels=True)
    plt.tight_layout()
    plt.show()

def make_heatmap(dfCPU, dfGPU, vmin=0, vmax=2000) -> None:
    def _plot(cgpu, vmin=vmin, vmax=vmax):
        sns.heatmap(cgpu, 
                    #annot=True, 
                    #fmt='.1f',
                    square=True,
                    vmin=vmin, vmax=vmax,
                    cbar=True,
                    xticklabels=True,
                    yticklabels=True)
        plt.tight_layout()
        plt.show()

    cpu = dfCPU.pivot_table(columns = 'POPULATION',
                            index   = 'CHROMOSOME',
                            values  = 'ELAPSEDTIME')[::-1]
    gpu = dfGPU.pivot_table(columns = 'POPULATION',
                            index   = 'CHROMOSOME',
                            values  = 'ELAPSEDTIME')[::-1]
    ratio = cpu / gpu

    _plot(cpu)
    _plot(gpu)
    _plot(ratio, vmin=0, vmax=10)

