#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import japanize_matplotlib


def load_meas_cond(fname):
    return pd.read_csv(fname)


def run():
    p = Path()
    w = p / "result.csv"
    
    paths = [path for path in p.glob("**/*") \
    if re.search(r"^\d{8}-\d{4}.*\.csv$", str(path))]

    with w.open('w') as fw:
        fw.write(','.join(['fname', 'datetime', 'r_tb', 'r_b', 'th_tb', 'th_b',
                           'cx', 'cy', 'cx_ft_removed', 'cy_ft_removed']) +
                 '\n')

        for path in paths:
            try:
                fname = path.name
                df = load_data(path, enable_path=True)
                date_time = df.index[0].strftime('%Y/%m/%d %H:%M:%S')
                gf, ptbx, ptby, pbx, pby = make_fit(df)
                r_tb, r_b, th_tb, th_b = get_rth(ptbx, ptby, pbx, pby)
                cx, cy, cx_ft_removed, cy_ft_removed = get_corr(gf)
                #make_plot(gf, fname=fname)
                csv_data = ','.join([str(fname), str(date_time), str(r_tb),
                                     str(r_b), str(th_tb), str(th_b), str(cx),
                                     str(cy), str(cx_ft_removed),
                                     str(cy_ft_removed)])
                fw.write(csv_data + '\n')

            except Exception as e:
                print(e)


def load_data(fname, enable_path=False):

    if not enable_path:
        p = Path(os.getcwd())
        fr = p / fname
    else:
        fr = fname

    df = pd.read_csv(fr, index_col=0, parse_dates=True)

    df = df.assign(time = df.index.astype(np.int64) // 10**9)
    df.time -= df.time[0]
    df = df.assign(baro_hpa = df.baro / 100)
    df = df.assign(delta_x = (df.x - df.x[0])*1000)
    df = df.assign(delta_y = (df.y - df.y[0])*1000)
    df = df.assign(delta_baro = df.baro_hpa - df.baro_hpa[0])

    return df


def make_fit(df):

    def __fitfunc_tb(X, offset, a, b, c, d):
        baro, time = X
        return offset + a*baro + b*time + c*time**2 + d*time**3

    def __fitfunc_b(x, offset, a):
        baro = x
        return offset + a*baro

    ptbx, ctbx = curve_fit(__fitfunc_tb, (df.delta_baro, df.time), df.delta_x)
    ptby, ctby = curve_fit(__fitfunc_tb, (df.delta_baro, df.time), df.delta_y)

    pbx, cbx = curve_fit(__fitfunc_b, df.delta_baro, df.delta_x)
    pby, cby = curve_fit(__fitfunc_b, df.delta_baro, df.delta_y)

    df = df.assign(x_fit_t = __fitfunc_tb((0, df.time), *ptbx)) 
    df = df.assign(y_fit_t = __fitfunc_tb((0, df.time), *ptby)) 
    df = df.assign(x_fit_b = __fitfunc_tb((df.delta_baro, 0), *ptbx)) 
    df = df.assign(y_fit_b = __fitfunc_tb((df.delta_baro, 0), *ptby)) 

    df = df.assign(delta_x_ft_removed = df.delta_x - df.x_fit_t)
    df = df.assign(delta_y_ft_removed = df.delta_y - df.y_fit_t)
    df = df.assign(delta_x_fb_removed = df.delta_x - df.x_fit_b)
    df = df.assign(delta_y_fb_removed = df.delta_y - df.y_fit_b)

    return df, ptbx, ptby, pbx, pby


def get_rth(ptbx, ptby, pbx, pby):
    slope_tb_x = ptbx[1]
    slope_tb_y = ptby[1]

    slope_b_x = pbx[1]
    slope_b_y = pby[1]

    r_tb = np.sqrt(slope_tb_x ** 2 + slope_tb_y ** 2)
    r_b  = np.sqrt(slope_b_x ** 2 + slope_b_y ** 2)

    # I'm not sure which one of following codes is better. Here I just adopted
    # upper one without clear reason.
    th_tb = np.rad2deg(np.arctan2(slope_tb_y, slope_tb_x))
    th_b  = np.rad2deg(np.arctan2(slope_b_y, slope_b_x))
    #th_tb = np.sign(slope_tb_y) * np.rad2deg(np.arctan(slope_tb_y / slope_tb_x))
    #th_b  = np.sign(slope_b_y) * np.rad2deg(np.arctan(slope_b_y / slope_b_x))

    return r_tb, r_b, th_tb, th_b


def get_corr(df):
    corr = df[['delta_x',
               'delta_y',
               'delta_x_ft_removed',
               'delta_y_ft_removed',
               'delta_baro']].corr(method='pearson')

    cx = corr.delta_baro.delta_x
    cy = corr.delta_baro.delta_y
    cx_ft_removed = corr.delta_baro.delta_x_ft_removed
    cy_ft_removed = corr.delta_baro.delta_y_ft_removed

    return cx, cy, cx_ft_removed, cy_ft_removed


def make_plot(df, fname="test"):

    fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(nrows=2, ncols=2,
                                                 sharex=False,
                                                 sharey=True,
                                                 figsize=(10, 10))

    axes = {
        0:{
            "ax":ax0,  # axUL
            "x":df.delta_baro,
            "y1":df.delta_x,
            "y2":df.delta_x_ft_removed,
            "color1":'red',
            "color2":'blue',
            "legend1":'生データX',
            "legend2":'X時間依存除去',
            "xlabel":'気圧[hPa]',
            "ylabel":'移動量[nm]',
            "xlim":[-30, 30],
            "ylim":[-60, 60]
        },
        1:{
            "ax":ax1,  # axUR
            "x":df.delta_baro,
            "y1":df.delta_y, 
            "y2":df.delta_y_ft_removed,
            "color1":'green',
            "color2":'y',
            "legend1":'生データY',
            "legend2":'Y時間依存除去',
            "xlabel":'気圧[hPa]',
            "ylabel":'移動量[nm]',
            "xlim":[-30, 30],
            "ylim":[-60, 60]
        },
        2:{
            "ax":ax2,  # axDL
            "x":df.time, 
            "y1":df.delta_x,
            "y2":df.delta_x_fb_removed,
            "color1":'red',
            "color2":'blue',
            "legend1":'生データX',
            "legend2":'X気圧依存除去',
            "xlabel":'Time[sec]',
            "ylabel":'移動量[nm]',
            "xlim":[0, df.time[-1]*1.1],
            "ylim":[-60, 60]
        },
        3:{
            "ax":ax3,  # axDR
            "x":df.time,
            "y1":df.delta_y, 
            "y2":df.delta_y_fb_removed,
            "color1":'green',
            "color2":'y',
            "legend1":'生データY',
            "legend2":'Y気圧依存除去',
            "xlabel":'Time[sec]',
            "ylabel":'移動量[nm]',
            "xlim":[0, df.time[-1]*1.1],
            "ylim":[-60, 60]
        }
    }

    for k, v in axes.items():
        # delta_x 圧力依存
        v["ax"].scatter(v["x"], v["y1"],
                     s=1, color=v["color1"], label=v["legend1"])
        v["ax"].scatter(v["x"], v["y2"],
                     s=1, color=v["color2"], label=v["legend2"])

        v["ax"].set_xlabel(v["xlabel"])
        v["ax"].set_ylabel(v["ylabel"])
        v["ax"].set_xlim(v["xlim"][0], v["xlim"][1])
        v["ax"].set_ylim(v["ylim"][0], v["ylim"][1])
        v["ax"].grid(ls='dashed', color='gray', alpha=0.5)
        v["ax"].legend()

    plt.suptitle(fname)
    plt.savefig(fname.split('.')[0] + ".png")
    plt.clf()

