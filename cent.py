#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_complex(a:complex, b:complex,
                 xymin=-0.04, xymax=0.04) -> None:
    x1, y1 = a.real, a.imag
    x2, y2 = b.real, b.imag

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.quiver(0, 0, x1, y1, color='green', alpha=0.8, linewidth=2)
    plt.quiver(0, 0, x2, y2, color='red', alpha=0.8, linewidth=2)

    plt.xlim(xymin, xymax)
    plt.ylim(xymin, xymax)

    plt.grid(ls='dashed', color='gray', alpha=0.5)
    plt.show()


def vector_and_array():
    a1 = np.array([0.00, 0.00])
    a2 = np.array([0.25, 0.00])
    a3 = np.array([0.25, 0.25])
    
    u1 = np.array([-0.064039, 0.046027])
    u2 = np.array([-0.095422, 0.049760])
    u3 = np.array([-0.069489, 0.015889])

    ma = u2 - u1
    mb = u3 - u1
    mc = a2 - a1
    md = a3 - a1

    print(ma, mb, mc, md)

    A = np.column_stack((ma, mb))
    B = np.column_stack((mc, md))
    Ainv = np.linalg.inv(A)
    print("A")
    print(A)
    print("Ainv")
    print(Ainv)
    print("B")
    print(B)

    C = B @ Ainv
    #C = Ainv @ B
    print("C")
    print(C)

    print("C@u1")
    print(C @ u1)


def main() -> None:

    # a1, a2, a3がパラメータで、u1, u2, u3はそれぞれa1, a2, a3
    # をとるときの値。

    # パラメータ
    a1 = 0.00 + 0.00j
    a2 = 0.25 + 0.00j
    a3 = 0.00 + 0.25j

    # 目的関数値
    u1 = -0.064039 + 0.046027j
    u2 = -0.095422 + 0.049760j
    u3 = -0.069489 + 0.015889j

    ma = u2 - u1
    mb = u3 - u1
    mc = a2 - a1
    md = a3 - a1

    #plot_complex(ma, mb, xymin=-0.03, xymax=0.03)
    #plot_complex(mc, md, xymin=-0.03, xymax=0.03)

    # u1を基準とした目的関数値の差分をとる。
    # それらを列ベクトルとした行列Aを定義する
    A = np.array([[ma.real, mb.real], [ma.imag, mb.imag]])
    print("A")
    print(A)
    # Aの逆行列を求める
    Ainv = np.linalg.inv(A)
    print("Ainv")
    print(Ainv)

    MA_real = Ainv[0, 0]
    MB_real = Ainv[0, 1]
    MA_imag = Ainv[1, 0]
    MB_imag = Ainv[1, 1]

    MA = complex(MA_real, MA_imag)
    MB = complex(MB_real, MB_imag)

    print("MA")
    print(MA)
    print("MB")
    print(MB)
    #plot_complex(MA, MB, xymin=-30, xymax=30)

    a = mc.real * MA.real + md.real * MA.imag
    b = mc.real * MB.real + md.real * MB.imag
    c = mc.imag * MA.real + md.imag * MA.imag
    d = mc.imag * MB.real + md.imag * MB.imag
    print("a, b, c, d")
    print(a, b, c, d)

    da = complex(a * u1.real + b * u1.imag,
                 c * u1.real + d * u1.imag)

    #plot_complex(da, a1 - da)
    print("da")
    print(da)
    #print(a1 - da)


if __name__ == '__main__':
    main()
    print("\n\n")
    vector_and_array()
