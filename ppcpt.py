#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load(fin:str) -> pd.DataFrame:
    names = ['w0', 'w1', 'num']
    df = pd.read_csv(fin, names=names)
    return df

def plot(df:pd.DataFrame) -> None:
    bins = range(df['num'].min(), df['num'].max() +2)
    df['num'].plot.hist(bins=bins, align='left')
    plt.grid(ls='dashed', which='major', axis='both', color='gray', alpha=0.3)
    plt.xticks(np.arange(0, 20))
    plt.yticks(np.arange(0, 110, 10))
    plt.show()

def hist(df, name="ref") -> None:
    fig, ax0 = plt.subplots(nrows=1, ncols=1)
    bins = range(df['num'].min(), df['num'].max() +2)
    df['num'].plot.hist(
        bins=bins, align='left', ax=ax0, alpha=0.5, label=name, ec='black')
    ax0.grid(ls='dashed', color='gray', alpha=0.3)
    ax0.set_xlabel("Iteration")
    ax0.set_xticks(np.arange(0, 20))
    ax0.set_yticks(np.arange(0, 110, 10))
    plt.legend()
    plt.show()

def cmp_hist2(df1, df2, name1="ref", name2="tgt") -> None:
    fig, ax0 = plt.subplots(nrows=1, ncols=1)
    bins = range(df1['num'].min(), df1['num'].max() +2)
    df1['num'].plot.hist(
        bins=bins, align='left', ax=ax0, alpha=0.5, label=name1, ec='black')
    df2['num'].plot.hist(
        bins=bins, align='left', ax=ax0, alpha=0.5, label=name2, ec='black')
    ax0.grid(ls='dashed', color='gray', alpha=0.3)
    ax0.set_xticks(np.arange(0, 20))
    ax0.set_yticks(np.arange(0, 110, 10))
    plt.legend()
    plt.show()

def cmp_hist3(df1, df2, df3) -> None:
    fig, ax0 = plt.subplots(nrows=1, ncols=1)
    bins = range(df1['num'].min(), df1['num'].max() +2)
    df1['num'].plot.hist(bins=bins, align='left', ax=ax0, alpha=0.5, ec='black', label='original')
    df2['num'].plot.hist(bins=bins, align='left', ax=ax0, alpha=0.5, ec='black', label='myModel1')
    df3['num'].plot.hist(bins=bins, align='left', ax=ax0, alpha=0.5, ec='black', label='myModel2')
    ax0.grid(ls='dashed', color='gray', alpha=0.3)
    ax0.set_xticks(np.arange(0, 20))
    ax0.set_yticks(np.arange(0, 110, 10))
    plt.legend()
    plt.show()

def heatmap1(df1, name1="ref") -> None:
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    reg = df1.pivot_table(columns="w0", index="w1", values="num")[::-1]
    sns.heatmap(reg, square=True, ax=ax0,
                cbar=True, cbar_kws={"shrink":0.5}, cmap='coolwarm',
                annot=True, vmin=0, vmax=15)
    ax0.set_title(name1)
    plt.tight_layout()
    plt.show()

def heatmap2(df1, df2, name1="ref", name2="tgt") -> None:
    fig, [ax0, ax1] = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))
    reg = df1.pivot_table(columns="w0", index="w1", values="num")[::-1]
    mym = df2.pivot_table(columns="w0", index="w1", values="num")[::-1]
    #sns.heatmap(reg, square=True)
    sns.heatmap(reg, square=True, ax=ax0,
                cbar=True, cbar_kws={"shrink":0.5}, cmap='coolwarm',
                annot=True, vmin=0, vmax=15)
    sns.heatmap(mym, square=True, ax=ax1,
                cbar=True, cbar_kws={"shrink":0.5}, cmap='coolwarm',
                annot=True, vmin=0, vmax=15)
    #sns.heatmap(reg, square=True, ax=ax0, vmin=0, vmax=15)
    #sns.heatmap(mym, square=True, ax=ax1, vmin=0, vmax=15)
    #sns.heatmap(reg, square=True, ax=ax0, cbar_ax=ax1, vmin=0, vmax=15)
    #sns.heatmap(mym, square=True, ax=ax1, cbar_ax=ax1, vmin=0, vmax=15)
    ax0.set_title(name1)
    ax1.set_title(name2)
    plt.tight_layout()
    plt.show()
