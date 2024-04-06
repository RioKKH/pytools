#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files1 = [
    "with_elitism_01.dat",
    "with_elitism_02.dat",
    "with_elitism_03.dat",
    "with_elitism_04.dat",
    "with_elitism_05.dat",
    "with_elitism_06.dat",
    "with_elitism_07.dat",
    "with_elitism_08.dat",
    "with_elitism_09.dat",
    "with_elitism_10.dat",
]

files2 = [
    "without_elitism_01.dat",
    "without_elitism_02.dat",
    "without_elitism_03.dat",
    "without_elitism_04.dat",
    "without_elitism_05.dat",
    "without_elitism_06.dat",
    "without_elitism_07.dat",
    "without_elitism_08.dat",
    "without_elitism_09.dat",
    "without_elitism_10.dat",
]

def load_data_CPU(fin:str) -> pd.DataFrame:
    names = ("gen", "FITMEAN", "FITMIN", "FITMAX", "FITSTD")

    df = pd.read_csv(fin, sep=",", names=names)
    return df

def load_data_GPU(fin:str) -> pd.DataFrame:
    names = ("gen", "POPULATION", "CHROMOSOME", "ELAPSEDTIME",
             "FITMAX", "FITMIN", "FITMEAN")
    df = pd.read_csv(fin, sep=",", names=names)
    return df

def load_data():
    header = ['gen', 'population', 'chromosome', 'elapsedtime',
              'fitness_max', 'fitness_min', 'fitness_mean']

    dfwith = [pd.read_csv(file, sep=",", header=None) for file in files1]
    dfwithout = [pd.read_csv(file, sep=",", header=None) for file in files2]
    #dfwith = [pd.read_csv(file, sep=",", header=header) for file in files1]
    #dfwithout = [pd.read_csv(file, sep=",", header=header) for file in files2]

    for df in dfwith:
        df.columns = ['gen', 'population', 'chromosome', 'elapsedtime',
                      'fitness_max', 'fitness_min', 'fitness_mean']

    for df in dfwithout:
        df.columns = ['gen', 'population', 'chromosome', 'elapsedtime',
                      'fitness_max', 'fitness_min', 'fitness_mean']

    avg_with_mean    = sum(df['fitness_mean'] for df in dfwith)    / len(dfwith)
    avg_without_mean = sum(df['fitness_mean'] for df in dfwithout) / len(dfwithout)
    avg_with_max     = sum(df['fitness_max']  for df in dfwith)    / len(dfwith)
    avg_without_max  = sum(df['fitness_max']  for df in dfwithout) / len(dfwithout)
    avg_with_min     = sum(df['fitness_min']  for df in dfwith)    / len(dfwith)
    avg_without_min  = sum(df['fitness_min']  for df in dfwithout) / len(dfwithout)

    df_avg_with = pd.DataFrame({'gen': dfwith[0]['gen'],
                                'fitness_mean': avg_with_mean,
                                'fitness_max': avg_with_max,
                                'fitness_min': avg_with_min})

    df_avg_without = pd.DataFrame({'gen': dfwithout[0]['gen'],
                                   'fitness_mean': avg_without_mean,
                                   'fitness_max': avg_without_max,
                                   'fitness_min': avg_without_min})

    return df_avg_with, df_avg_without

def plot(df_avg_with, df_avg_without):
    plt.figure(figsize=(12, 6))
    plt.plot(df_avg_with['gen'], df_avg_with['fitness_mean'],
             label='avg_with', alpha=0.8)
    plt.fill_between(df_avg_with['gen'], 
                     df_avg_with['fitness_min'], df_avg_with['fitness_max'],
                     color='gray', alpha=0.3)

    plt.plot(df_avg_without['gen'], df_avg_without['fitness_mean'],
             label='avg_without', alpha=0.8)
    plt.fill_between(df_avg_without['gen'],
                     df_avg_without['fitness_min'], df_avg_without['fitness_max'],
                     color='gray', alpha=0.3)

    plt.title("Fitness Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(ls="dashed", color="gray", alpha=0.5)
    plt.legend(loc="upper right")
    plt.show()


def plot_cpu_vs_gpu(df1, df2, label1="CPU_MEAN", label2="GPU_MEAN"):
    plt.figure(figsize=(12, 6))
    plt.plot(df1['gen'], df1['FITMEAN'], label=label1, alpha=0.8)
    plt.fill_between(df1['gen'], df1['FITMIN'], df1['FITMAX'], alpha=0.3)

    plt.plot(df2['gen'], df2['FITMEAN'], label=label2, alpha=0.8)
    plt.fill_between(df2['gen'], df2['FITMIN'], df2['FITMAX'], alpha=0.3)

    plt.title("Fitness Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(ls="dashed", color="gray", alpha=0.5)
    plt.legend(loc="upper right")
    plt.show()




