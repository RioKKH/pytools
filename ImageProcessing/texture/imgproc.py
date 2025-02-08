#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt


def im2col_simple(image, flt_h, flt_w, out_h, out_w):
    """
    チャンネル=1, バッチサイズ=1のimg2col
    """

    # 入力画像の高さ、幅
    img_h, img_w = image.shape

    # 生成される行列のサイズ
    cols = np.zeros((flt_h*flt_w, out_h*out_w))

    for h in range(out_h): # 出力画像の高さでforループを回す
        h_lim = h + flt_h
        for w in range(out_w): # 出力画像の幅でforループを回す
            w_lim = w + flt_w
            # 入力画像からフィルタのサイズの領域をスライスする
            # スライスした領域をrehape(-1)で平坦にし、生成される行列
            # colsの列に代入する
            cols[:, h*out_w+w] = image[h:h_lim, w:w_lim].reshape(-1)

    return cols


def im2col_fast(image, flt_h, flt_w, out_h, out_w):

    img_h, img_w = image.shape
    # ここでcolsは(Fh, Fw, Oh, Ow)形状の4次元配列になる
    cols = np.zeros((flt_h, flt_w, out_h, out_w))

    for h in range(flt_h):
        h_lim = h + out_h
        for w in range(flt_w):
            w_lim = w + out_w
            # 4次元配列のcolsに入力画像の領域がスライスされて格納される
            cols[h, w, :, :] = image[h:h_lim, w:w_lim]

    # reshapeすることで(FhFw, OhOw)の配列になる
    cols = cols.reshape(flt_h*flt_w, out_h*out_w)

    return cols


def im2col(images, flt_h, flt_w, out_h, out_w, stride, pad):

    n_bt, n_ch, img_h, img_w = images.shape

    # paddingを実装している箇所
    # padはpaddingの幅。配列imagesにはパッチ、チャンネルごとの画像が入っている
    # 下のように書く事で、入力画像の上下左右にのみ0が挿入された配列を得ることが出来る
    # 最後の引数を"constant"にすることで、同じ値を挿入することが出来る
    img_pad = np.pad(images,
                        [(0, 0), (0, 0), (pad, pad), (pad, pad)],
                        "constant")

    # 配列colsは最初に(B, C, Fh, Fw, Oh, Ow)の6次元配列(6階のテンソル)
    cols = np.zeros((n_bt, n_ch, flt_h, flt_w, out_h, out_w))

    for h in range(flt_h):
        h_lim = h + stride*out_h
        for w in range(flt_w):
            w_lim = w + stride*out_w

            # フィルタのピクセル事に割り当てられるimagesの領域がスライスされる
            cols[:, :, h, w, :, :] = img_pad[:, :,
                                                h:h_lim:stride,
                                                w:w_lim:stride]

    # transpose関数によって(C, Fh, Fw, B, Oh, Ow)と軸を入れ替える
    # これをreshapeすることで形状が(CFhFw, BOhOw)の行列が得られる
    cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(
        n_ch*flt_h*flt_w, n_bt*out_h*out_w)

    return cols
                                        

def col2im(cols, img_shape, flt_h, flt_w, out_h, out_w, stride, pad):

    # 引数として受け取る行列colsｎ形状は(CFhFw, BOhOw)
    n_bt, n_ch, img_h, img_w = img_shape

    # colsを6次元配列に分解したうえで、軸を入れ替える
    cols = cols.reshape(n_ch, flt_h, flt_w, n_bt, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)
    # 上記処理により配列の形状は(B, C, Fh, Fw, Oh, Ow)となる

    # 返還後の画像を格納する4次元配列を生成する
    # padding文だけ高さと幅を追加し、また、画像がストライドの値で割り切れない場合を考慮して
    # stride-1を幅と高さに加えておく。
    images = np.zeros((n_bt, n_ch, img_h+2*pad+stride-1, img_w+2*pad+stride-1))

    for h in range(flt_h):
        h_lim = h + stride*out_h
        for w in range(flt_w):
            w_lim = w + stride*out_w
            # imagesにcolsにおけるフィルタの該当箇所を格納する
            images[:, :, h:h_lim:stride,
                    w:w_lim:stride] += cols[:, :, h, w, :, :]

    # 最後にパディング分をスライスにより取り除くと画像への変換は完了
    return images[:, :, pad:img_h+pad, pad:img_w+pad]


def col2im_fast(cols, img_shape, flt_h, flt_w, out_h, out_w):

    # 引数として受け取る行列colsの形状は(FhFw, OhOw)
    img_h, img_w = img_shape

    cols = cols.reshape(flt_h, flt_w, out_h, out_w).transpose(0, 1, 2, 3)
    images = np.zeros((img_h, img_w))

    for h in range(flt_h):
        h_lim = h + out_h
        for w in range(flt_w):
            w_lim = w + out_w
            images[h:h_lim, w:w_lim] += cols[:, :, h, w]

    return images

