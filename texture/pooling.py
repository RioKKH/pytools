#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from imgproc import im2col, col2im

class Pooling():
    """
    n_bt : batch size
    x_ch : number of channel for input
    pool : size of pooling area
    pad  : size of padding
    y_ch : number of channel for output
    y_h  : height of output
    y_w  : width of output
    """

    def __init__(self, x_ch, x_h, x_w, pool, pad):

        # パラメータをまとめる
        self.params = (x_ch, x_h, x_w, pool, pad)

        self.y_ch = x_ch
        self.y_h = x_h//pool if x_h%pool==0 else x_h//pool+1
        self.y_w = x_w//pool if x_w%pool==0 else x_w//pool+1

    def pooling(self, x, kind="max"):
        n_bt = x.shape[0]
        x_ch, x_h, x_w, pool, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        # 入力画像を行列に変換
        cols = im2col(x, pool, pool, y_h, y_w, pool, pad)
        # --> colsの形状は(CPP, BOhOw)
        cols = cols.T.reshape(n_bt*y_h*y_w*y_ch, pool*pool)
        # --> colsの形状は(BOhOwC, PP)
        print(cols)

        # 出力の計算
        if kind == "max":
            y = np.max(cols, axis=1)
        elif kind == "min":
            y = np.min(cols, axis=1)
        elif kind == "average":
            y = np.average(cols, axis=1)
        # --> yの形状はcolsから次元が1つ減って(BOhOwC)

        self.y = y.reshape(n_bt, y_h, y_w, y_ch).transpose(0, 3, 1, 2)
        # --> (B, C, Oh, Ow)

        #ret_cols = y.reshape(n_bt, y_h, y_w, y_ch)
        ret_cols = y.reshape(1, 1, n_bt, y_h, y_w, y_ch)
        #ret_cols = y.reshape(pool, pool, n_bt, y_h, y_w, y_ch)
        #ret_cols = ret_cols.reshape(pool, pool, n_bt, y_h, y_w, y_ch)
        # --> (P, P, B, Oh, Ow, C)

        ret_cols = ret_cols.transpose(5, 0, 1, 2, 3, 4)
        # --> (C, P, P, Oh, Ow, C)

        ret_cols = ret_cols.reshape(y_ch*1*1, n_bt*y_h*y_w)
        #ret_cols = ret_cols.reshape(y_ch*pool*pool, n_bt*y_h*y_w)
        # --> (CPP, BOhOw)

        x_shape = (n_bt, x_ch, 2, 2)
        #x_shape = (n_bt, x_ch, x_h, x_w)

        ret_cols = col2im(ret_cols, x_shape, 1, 1, y_h, y_w, 1, pad)
        #return col2im(ret_cols, x_shape, pool, pool, y_h, y_w, pool, pad)
        return ret_cols[0, 0, :, :]

