#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path


def load(filename:str) -> pd.DataFrame:
    # Load data
    df = pd.read_csv(filename)
    return df


def plot(#dfcpu:pd.DataFrame,
         dfgpu_pe:pd.DataFrame,
         dfgpu_re:pd.DataFrame,
         pop_size, chrom_size) -> None:
    # Plot
    fig, ax = plt.subplots()
    #dfcpu.plot(x='generation', y='mean', ax=ax, label='CPU')
    dfgpu_re.plot(x='generation', y='mean', ax=ax, label='GPU regular elitism')
    dfgpu_pe.plot(x='generation', y='mean', ax=ax, label='GPU pesudo elitism')
    #plt.fill_between(dfcpu['generation'], dfcpu['min'], dfcpu['max'], alpha=0.3)
    plt.fill_between(dfgpu_re['generation'], dfgpu_re['min'], dfgpu_re['max'], alpha=0.3)
    plt.fill_between(dfgpu_pe['generation'], dfgpu_pe['min'], dfgpu_pe['max'], alpha=0.3)
    plt.ylim(0, chrom_size)
    plt.show()

def plotNyoro(dfgpu_pe: pd.DataFrame,
              dfgpu_re: pd.DataFrame,
              pop_size: int,
              chrom_size: int,
              show=False) -> None:

    # フォントサイズの設定
    plt.rcParams.update({'font.size': 14})

    # 上下の比率を3:1に設定
    fig, ax = plt.subplots(nrows=2, figsize=(7, 5), sharex='col',
                           gridspec_kw={'height_ratios': [3, 1]})
    # グラフの背景色を白に設定する
    fig.patch.set_facecolor('white')
    # 上部と下部に同じデータを描画する
    #dfcpu.plot(x='generation', y='mean', ax=ax[0], label='CPU')
    dfgpu_re.plot(x='generation', y='mean', ax=ax[0], label='Regular Elitism')
    dfgpu_pe.plot(x='generation', y='mean', ax=ax[0], label='Pesudo  Elitism')
    #plt.fill_between(dfcpu['generation'], dfcpu['min'],
    #ax[0].fill_between(dfcpu['generation'], dfcpu['min'],
    #                 dfcpu['max'], alpha=0.3)#, ax=ax[0])
    #plt.fill_between(dfgpu_re['generation'], dfgpu_re['min'],
    ax[0].fill_between(dfgpu_re['generation'], dfgpu_re['min'],
                     dfgpu_re['max'], alpha=0.3)#, ax=ax[0])
    #plt.fill_between(dfgpu_pe['generation'], dfgpu_pe['min'],
    ax[0].fill_between(dfgpu_pe['generation'], dfgpu_pe['min'],
                     dfgpu_pe['max'], alpha=0.3)#, ax=ax[0])

    #dfcpu.plot(x='generation', y='mean', ax=ax[1], label='CPU')
    dfgpu_re.plot(x='generation', y='mean', ax=ax[1], label='Regular Elitism')
    dfgpu_pe.plot(x='generation', y='mean', ax=ax[1], label='Pesudo  Elitism')
    #plt.fill_between(dfcpu['generation'], dfcpu['min'],
    #ax[1].fill_between(dfcpu['generation'], dfcpu['min'],
    #                 dfcpu['max'], alpha=0.3)#, ax=ax[1])
    #plt.fill_between(dfgpu_re['generation'], dfgpu_re['min'],
    ax[1].fill_between(dfgpu_re['generation'], dfgpu_re['min'],
                     dfgpu_re['max'], alpha=0.3)#, ax=ax[1])
    #plt.fill_between(dfgpu_pe['generation'], dfgpu_pe['min'],
    ax[1].fill_between(dfgpu_pe['generation'], dfgpu_pe['min'],
                     dfgpu_pe['max'], alpha=0.3)#, ax=ax[1])

    # 上部の凡例位置を上に調整する
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.4, 0.7), fontsize=18)

    # 下部の凡例を非表示にする
    ax[1].get_legend().remove()

    # サブプロット間の上下間隔をゼロに設定する
    fig.subplots_adjust(hspace=0.0)

    # 下段サブプロットの設定
    ax[1].set_ylim(0, (chrom_size/8)*2)
    ax[1].set_yticks(np.arange(0, chrom_size/8 + 1, chrom_size/8))
    print(np.arange(0, chrom_size/8 + 1, chrom_size/8))

    # 上段サブプロットの設定
    ax[0].set_ylim((chrom_size/8)*3, (chrom_size/8)*9)
    ax[0].set_yticks(np.arange((chrom_size/8)*(3+1), (chrom_size/8)*9 + 1, chrom_size/8))
    print(np.arange((chrom_size/8)*(3+1), (chrom_size/8)*9 + 1, chrom_size/8))

    # 下段のプロット領域上辺を非表示
    ax[1].spines['top'].set_visible(False)

    # 上段のプロット領域底辺を非表示、X軸のメモリとラベルを非表示
    ax[0].spines['bottom'].set_visible(False)
    ax[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # 上段と下段のＹ軸の目盛りの設定
    #ax[1].set_yticks(np.arange(0, (chrom_size/8)*2, chrom_size/8))
    #ax[1].set_yticks(np.arange(0, (chrom_size/8)*2 + 1, chrom_size/8))
    #ax[0].set_yticks(np.arange((chrom_size/8)*3, (chrom_size/8)*9 + 1, chrom_size/8))

    # 下段のX軸の目盛りのフォントサイズとラベルのサイズを設定
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    ax[1].set_xlabel('Generation', fontsize=18)

    # 上段のY軸の目盛りのフォントサイズを設定
    ax[0].tick_params(axis='y', labelsize=16)

    d1 = 0.02 # X軸のはみだし量
    d2 = 0.03 # ニョロ線の高さ
    wn = 21   # ニョロの数(奇数値を指定すること)

    pp = (0, d2, 0, -d2)
    px = np.linspace(-d1, 1+d1, wn)
    py = np.array([1+pp[i%4] for i in range(0, wn)])
    p = Path(list(zip(px, py)), [Path.MOVETO] + [Path.CURVE3]*(wn-1))

    line1 = mpatches.PathPatch(p, lw=4, edgecolor='black',
                               facecolor='None', clip_on=False,
                               transform=ax[1].transAxes, zorder=10)
    line2 = mpatches.PathPatch(p, lw=3, edgecolor='white',
                               facecolor='None', clip_on=False,
                               transform=ax[1].transAxes, zorder=10,
                               capstyle='round')

    ax[0].grid(ls='--', alpha=0.5)
    ax[1].grid(ls='--', alpha=0.5)

    plt.suptitle(f"pop_size={pop_size}, chrom_size={chrom_size}",
                 fontsize=20)
    #plt.tight_layout()

    a = ax[1].add_patch(line1)
    a = ax[1].add_patch(line2)

    if show:
        plt.show()
    else:
        plt.savefig(f"fitnesstrend_{pop_size}_{chrom_size}.png")
        plt.close()


def makeFitSlide(show=True) -> None:
    gpure128 = (f"fitnesstrend_20240515-235139_128_128_avg_GPU.csv")
    gpupe128 = (f"fitnesstrend_20240516-155729_128_128_avg_GPU.csv")
    dfgpu_re128 = load(gpure128)
    dfgpu_pe128 = load(gpupe128)
    plotNyoro(dfgpu_pe128, dfgpu_re128, 128, 128, show=show)
    plt.close()

    #gpure1024 = (f"fitnesstrend_20240425-011943_1024_1024_avg_GPU.csv")
    #gpupe1024 = (f"fitnesstrend_20240425-010644_1024_1024_avg_GPU.csv")
    gpure1024 = (f"fitnesstrend_20240515-235139_1024_1024_avg_GPU.csv")
    gpupe1024 = (f"fitnesstrend_20240516-155729_1024_1024_avg_GPU.csv")
    dfgpu_re1024 = load(gpure1024)
    dfgpu_pe1024 = load(gpupe1024)
    plotNyoro(dfgpu_pe1024, dfgpu_re1024, 1024, 1024, show=show)
    plt.close()

def run(cpu_tgt: str, 
        gpu_pe_tgt: str,
        gpu_re_tgt: str,
        show=False) -> None:
    #for pop_size in range(1024, 1024+1, 128):
    #    for chrom_size in range(1024, 1024+1, 128):
    for pop_size in range(128, 1024+1, 128):
        for chrom_size in range(128, 1024+1, 128):
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
            #plot(dfcpu, dfgpu_pe, dfgpu_re, pop_size, chrom_size)
            plotNyoro(dfcpu, dfgpu_pe, dfgpu_re, pop_size, chrom_size, show=show)
