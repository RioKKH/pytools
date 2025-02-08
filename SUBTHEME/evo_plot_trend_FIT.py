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


def plotNyoro(show=True) -> None:

    # フォントサイズの設定
    plt.rcParams.update({'font.size': 14})

    # 上下の比率を3:1に設定
    fig, ax = plt.subplots(nrows=2, figsize=(7, 5), sharex='col',
                           gridspec_kw={'height_ratios': [3, 1]})
    # グラフの背景色を白に設定する
    fig.patch.set_facecolor('white')

    #pop_size = 128
    chrom_size = 1024

    # (128, 128)の場合のデータを読み込む
    gpure128 = (f"fitnesstrend_20240425-011943_128_1024_avg_GPU.csv")
    gpupe128 = (f"fitnesstrend_20240425-010644_128_1024_avg_GPU.csv")

    # (512, 512)の場合のデータを読み込む
    # gpure512 = (f"fitnesstrend_20240425-011943_512_1024_avg_GPU.csv")
    # gpupe512 = (f"fitnesstrend_20240425-010644_512_1024_avg_GPU.csv")

    # (1024, 1024)の場合のデータを読み込む
    gpure1024 = (f"fitnesstrend_20240425-011943_1024_1024_avg_GPU.csv")
    gpupe1024 = (f"fitnesstrend_20240425-010644_1024_1024_avg_GPU.csv")

    # データを読み込む
    dfgpu128_re = pd.read_csv(gpure128)
    dfgpu128_pe = pd.read_csv(gpupe128)
    # dfgpu512_re = pd.read_csv(gpure512)
    # dfgpu512_pe = pd.read_csv(gpupe512)
    dfgpu1024_re = pd.read_csv(gpure1024)
    dfgpu1024_pe = pd.read_csv(gpupe1024)

    # グレースケール用のカラーマップを設定
    plt.style.use('grayscale')

    # 上部と下部に同じデータを描画する
    dfgpu128_re.plot(x='generation', y='mean', ls='--', lw=2, alpha=0.5, ax=ax[0], label='GPU regular elitism (128, 128)')
    dfgpu128_pe.plot(x='generation', y='mean', ls='-', ax=ax[0], marker='o', markevery=32, label='GPU pesudo elitism (128, 128)')
    # dfgpu512_re.plot(x='generation', y='mean', ls='--', lw=2, ax=ax[0], label='GPU regular elitism (512, 512)')
    # dfgpu512_pe.plot(x='generation', y='mean', ls='-', ax=ax[0], label='GPU pesudo elitism (512, 512)')
    dfgpu1024_re.plot(x='generation', y='mean', ls='--', lw=2, alpha=0.5, ax=ax[0], label='GPU regular elitism (1024, 1024)')
    dfgpu1024_pe.plot(x='generation', y='mean', ls='-', ax=ax[0], marker='^', markevery=32, label='GPU pesudo elitism (1024, 1024)')

    #ax[0].fill_between(dfgpu128_re['generation'], dfgpu128_re['min'], dfgpu128_re['max'], alpha=0.3)
    #ax[0].fill_between(dfgpu128_pe['generation'], dfgpu128_pe['min'], dfgpu128_pe['max'], alpha=0.3)
    # ax[0].fill_between(dfgpu512_re['generation'], dfgpu512_re['min'], dfgpu512_re['max'], alpha=0.3)
    # ax[0].fill_between(dfgpu512_pe['generation'], dfgpu512_pe['min'], dfgpu512_pe['max'], alpha=0.3)
    #ax[0].fill_between(dfgpu1024_re['generation'], dfgpu1024_re['min'], dfgpu1024_re['max'], alpha=0.3)
    #ax[0].fill_between(dfgpu1024_pe['generation'], dfgpu1024_pe['min'], dfgpu1024_pe['max'], alpha=0.3)

    dfgpu128_re.plot(x='generation', y='mean', ls='--', lw=3, alpha=0.5, ax=ax[1], label='GPU regular elitism (128, 128)')
    dfgpu128_pe.plot(x='generation', y='mean', ls='-', ax=ax[1], label='GPU pesudo elitism (128, 128)')
    # dfgpu512_re.plot(x='generation', y='mean', ls='--', lw=2, ax=ax[1], label='GPU regular elitism (512, 512)')
    # dfgpu512_pe.plot(x='generation', y='mean', ls='-', ax=ax[1], label='GPU pesudo elitism (512, 512)')
    dfgpu1024_re.plot(x='generation', y='mean', ls='-.', lw=3, alpha=0.5, ax=ax[1], label='GPU regular elitism (1024, 1024)')
    dfgpu1024_pe.plot(x='generation', y='mean', ls=':', ax=ax[1], label='GPU pesudo elitism (1024, 1024)')

    #ax[1].fill_between(dfgpu128_re['generation'], dfgpu128_re['min'], dfgpu128_re['max'], alpha=0.3)
    #ax[1].fill_between(dfgpu128_pe['generation'], dfgpu128_pe['min'], dfgpu128_pe['max'], alpha=0.3)
    # ax[1].fill_between(dfgpu512_re['generation'], dfgpu512_re['min'], dfgpu512_re['max'], alpha=0.3)
    # ax[1].fill_between(dfgpu512_pe['generation'], dfgpu512_pe['min'], dfgpu512_pe['max'], alpha=0.3)
    #ax[1].fill_between(dfgpu1024_re['generation'], dfgpu1024_re['min'], dfgpu1024_re['max'], alpha=0.3)
    #ax[1].fill_between(dfgpu1024_pe['generation'], dfgpu1024_pe['min'], dfgpu1024_pe['max'], alpha=0.3)

    # 上部の凡例位置を上に調整する
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.15, 0.65), fontsize=16)

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
    ax[0].set_ylabel('Fitness', fontsize=18)

    # 下段のX軸の目盛りのフォントサイズとラベルのサイズを設定
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    ax[1].set_xlabel('Generation', fontsize=18)
    #ax[1].set_ylabel('Fitness', fontsize=18)

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

    a = ax[1].add_patch(line1)
    a = ax[1].add_patch(line2)

    #plt.tight_layout()

    #plt.suptitle(f"chrom_size={chrom_size}", fontsize=20)

    if show:
        plt.show()
    else:
        plt.savefig(f"fitnesstrend_{chrom_size}.png")
        plt.close()


