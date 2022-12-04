#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt


class imgproc():

    def __init__(self):
        pass

    def binarize(self, img):
        pass

    def pooling_proto(self, img, k, kind="average"):
        _pooling = {
            "max": np.max,
            "min": np.min,
            "avg": np.average,
        }

        if self.color:
            width, height, color = img.shape
        else:
            width, height = img.shape

        size = k // 2
        dst = img.copy()

        for x in range(size, width, k):
            for y in range(size, height, k):
                if img.color:
                    dst[x-size:x+size, y-size:y+size, 0] = _pooling[kind](img[x-size:x+size, y-size:y+size, 0])
                    dst[x-size:x+size, y-size:y+size, 1] = _pooling[kind](img[x-size:x+size, y-size:y+size, 1])
                    dst[x-size:x+size, y-size:y+size, 2] = _pooling[kind](img[x-size:x+size, y-size:y+size, 2])

                else:
                    dst[x-size:x+size, y-size:y+size] = _pooling[kind](img[x-size:x+size, y-size:y+size])

        img.dst = dst


    def im2col_simple(self, image, flt_h, flt_w, out_h, out_w):
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


    def im2col_fast(self, image, flt_h, flt_w, out_h, out_w):

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


    def im2col(self, images, flt_h, flt_w, out_h, out_w, stride, pad):

        n_bt, n_ch, img_h, img_w = images.shape

        # paddingを実装している箇所
        # padはpaddingの幅。配列imagesにはパッチ、チャンネルごとの画像が入っている
        # 下のように書く事で、入力画像の上下左右にのみ0が挿入された配列を得ることが出来る
        # 最後の引数を"constant"にすることで、同じ値を挿入することが出来る
        img_pad = np.pad(images,
                         [(0, 0), (0, 0), (pad, pad), (pad, pad)],
                         "constant")

        cols = np.zeros((n_bt, n_ch, flt_h, flt_w, out_h, out_w))

        for h in range(flt_h):
            h_lim = h + stride*out_h
            for w in range(flt_w):
                w_lim = w + stride*out_w
                cols[:, :, h, w, :, :] = img_pad[:, :, h:h_lim:stride,
                                                 w:w_lim:stride]

        cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(
            n_ch*flt_h*flt_w, n_bt*out_h*out_w)

        return cols
                                         

    def convolution(self, imgfilter=None):
        pass

