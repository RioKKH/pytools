#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


def load(filename:str) -> pd.DataFrame:
    # Load data
    df = pd.read_csv(filename)
    return df


def plot(dfcpu:pd.DataFrame,
         dfgpu_pe:pd.DataFrame,
         dfgpu_re:pd.DataFrame) -> None:
    # Plot
    fig, ax = plt.subplots()
    #dfcpu.plot(x='generation', y='mean', ax=ax, label='CPU')
    dfgpu_re.plot(x='generation', y='mean', ax=ax, label='GPU regular elitism')
    #dfgpu_pe.plot(x='generation', y='mean', ax=ax, label='GPU pesudo elitism')
    #plt.fill_between(dfcpu['generation'], dfcpu['min'], dfcpu['max'], alpha=0.3)
    plt.fill_between(dfgpu_re['generation'], dfgpu_re['min'], dfgpu_re['max'], alpha=0.3)
    #plt.fill_between(dfgpu_pe['generation'], dfgpu_pe['min'], dfgpu_pe['max'], alpha=0.3)
    plt.show()

def run(cpu_tgt:str, gpu_pe_tgt:str, gpu_re_tgt:str) -> None:
    for pop_size in range(1024, 1024+1, 128):
        for chrom_size in range(1024, 1024+1, 128):
    #for pop_size in ranage(128, 1024+1, 128):
    #    for chrom_size in range(128, 1024+1, 128):
            cpufile = (f"fitnesstrend_{cpu_tgt}"
                       f"_{pop_size}_{chrom_size}_avg.csv")
            gpupefile = (f"fitnesstrend_{gpu_re_tgt}"
                         f"_{pop_size}_{chrom_size}_avg_GPU.csv")
            gpurefile = (f"fitnesstrend_{gpu_re_tgt}"
                         f"_{pop_size}_{chrom_size}_avg_GPU.csv")
            # Load data
            dfcpu = load(cpufile)
            dfgpu_pe = load(gpupefile)
            dfgpu_re = load(gpurefile)
            # Plot
            plot(dfcpu, dfgpu_pe, dfgpu_re)
