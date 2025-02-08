#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
以下のコードを写経させて頂いた
https://qiita.com/code0327/items/820cc9e239736ed33fdd
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

left = np.array(['fukushima', 'aichi', 'kanagawa', 'osaka', 'tokyo'])
height = np.array([160, 220, 280, 360, 1820])

# gridspec_kw={'height_ratios': (1, 2)} で上下のグラフの比率を指定
# (1, 2) で上下の比率を 1:2 に指定
fig, ax = plt.subplots(nrows=2, figsize=(3, 4), dpi=160, sharex='col',
                       gridspec_kw={'height_ratios': (1, 2)})

# グラフの背景色を白に設定
fig.patch.set_facecolor('white')

# 上部を下部に同じデータを描画する
ax[0].bar(left, height)
ax[1].bar(left, height)

# サブプロット間の上下感覚をゼロに設定
fig.subplots_adjust(hspace=0.0)

# 下段サブプロット
ax[1].set_ylim(0, 400) # 区間幅 400
ax[1].set_yticks(np.arange(0, 300+1, 100))

# 上段サブプロット
# height_ratios で上下の比率を1:2に指定しているため
# set_ylimでしているする区間幅も1:2になるようにする。
ax[0].set_ylim(1750, 1950) # 区間幅 200
ax[0].set_yticks((1800, 1900))

# 下段のプロット領域上辺を非表示
# spine(スパイン)とはグラフの周囲を囲む枠線のこと
# spine: 脊椎、背骨
ax[1].spines['top'].set_visible(False)

# 上段のプロット領域底辺を非表示、X軸のメモリとラベルを非表示
ax[0].spines['bottom'].set_visible(False)
ax[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

d1 = 0.02 # Ｘ軸のはみだし量
d2 = 0.03 # ニョロ戦の高さ
wn = 21   # ニョロ波の数(奇数地を指定)

pp = (0, d2, 0, -d2)
px = np.linspace(-d1, 1+d1, wn)
py = np.array([1+pp[i%4] for i in range(0, wn)])
# Pathクラスはカスタムの図形を描画するためのクラス
# これを用いることで、直線、曲線、複雑な図形を描画することができる
# 今回は、ニョロ波を描画するためにPathクラスを使用している
# class matplotlib.path.Path(
#     vertices,     # 図形の頂点の座標(リストの各要素は(x, y)のタプル)
#     codes=None,   # 図形の描画方法(リスト)
#     _interpolation_steps=1,
#     closed=False) # 図形が閉じているかどうか (True: 閉じている, False: 閉じていない)
p = Path(list(zip(px, py)), [Path.MOVETO] + [Path.CURVE3]*(wn-1))


# PathPatchクラスはPathオブジェクトを描画する為のパッチ(図形)を定義する
# class matplotlib.patches.PathPatch(path, **kwargs)
#     path: Pathオブジェクト
#     kwargs: その他の描画属性(線の幅、色、透明度など)
line1 = mpatches.PathPatch(p, lw=4, edgecolor='black',
                           facecolor='None', clip_on=False,
                           transform=ax[1].transAxes, zorder=10)
line2 = mpatches.PathPatch(p, lw=3, edgecolor='white',
                           facecolor='None', clip_on=False,
                           transform=ax[1].transAxes, zorder=10,
                           capstyle='round')
a = ax[1].add_patch(line1)
a = ax[1].add_patch(line2)

plt.show()
