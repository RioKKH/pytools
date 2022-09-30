#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

def load(fin:str) -> pd.DataFrame:
    df = pd.read_csv(fin)
    return df

def plot_scatter(df:pd.DataFrame, drange=1000) -> None:
    df[:drange].plot(marker='o',
            alpha=0.3)
    plt.grid(ls='dashed', color='gray', alpha=0.5)
    plt.show()

def plot_hist(df:pd.DataFrame, nbins=10) -> None:
    df.hist(bins=nbins)
    plt.show()

