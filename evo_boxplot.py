#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evo_elitism import EvoElitism

def load_data(fname:str) -> pd.DataFrame:
    df = EvoElitism(fname)
    df.make_average()
    return df

def plot_boxplot(df1:pd.DataFrame, df2:pd.DataFrame, title:str):
    """
    df1: DataFrame with regular elitism data
    df2: DataFrame with pseudo elitism data
    title: Title of the plot
    """

    regular = df1.heatmap_data.melt()['value']
    pseudo = df2.heatmap_data.melt()['value']
    diff = (df1.heatmap_data - df2.heatmap_data).melt()['value']

    fig, ax = plt.subplots()
    # sns.boxplot(data=combined, palette='Set3', ax=ax)
    # sns.boxplot(x='Type', y='value', data=combined, palette='Set3')
    sns.boxplot(data=[regular, pseudo, diff], palette='Greys', ax=ax)
    #ax.boxplot([regular, pseudo, diff],
    #           labels=['Regular Elitism', 'Pseudo Elitism', 'Difference'])

    ax.set_title(title, fontsize=20)
    ax.set_ylim(0, 20)
    ax.set_ylabel('Elapsed Time [msec]', fontsize=15)
    ax.set_xticklabels(['Regular Elitism', 'Pseudo Elitism', 'Difference'],
                       fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    #ax.set_ylabel('Elapsed Time [msec]')
    plt.show()

def main():
    df1 = load_data('regular_elite.csv')
    df2 = load_data('pseudo_elite.csv')
    plot_boxplot(df1, df2, '')
