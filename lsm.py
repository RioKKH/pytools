#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(fname):
    return pd.read_csv(fname, index_col=0, comment='#')

def make_matrix(df):
    a6x = df.a6x.values
    a6y = df.a6y.values
    real = df.real.values
    imag = df.imag.values
    G, y = __make_matrix_jit(a6x, a6y, real, imag)
    
    return G, y


@jit('(f8[:], f8[:], f8[:], f8[:])', nopython=True)
def __make_matrix_jit(a6x, a6y, real, imag):
    sum_xi2 = (a6x**2).sum()
    sum_yi2 = (a6y**2).sum()
    sum_xiyi = (a6x * a6y).sum() 
    sum_xi = a6x.sum()
    sum_yi = a6y.sum()

    sum_xiXi = (a6x * real).sum()
    sum_yiXi = (a6y * real).sum()
    sum_xiYi = (a6x * imag).sum()
    sum_yiYi = (a6y * imag).sum()
    sum_Xi = real.sum()
    sum_Yi = imag.sum()

    G = np.array([[sum_xi2,  sum_xiyi, 0,       0,        sum_xi,  0      ],
                  [sum_xiyi, sum_yi2,  0,       0,        sum_yi,  0      ],
                  [0,        0,        sum_xi2, sum_xiyi, 0,       sum_xi ],
                  [0,        0,        sum_xiyi,sum_yi2,  0,       sum_yi ],
                  [sum_xi,   sum_yi,   0,       0,        len(a6x), 0      ],
                  [0,        0,        sum_xi,  sum_yi,   0,       len(a6x)]])

    y = np.array([sum_xiXi, sum_yiXi, sum_xiYi, sum_yiYi, sum_Xi, sum_Yi])

    return G, y


@jit('(f8[:,:], f8[:])', nopython=True)
def solve(G, y):
    A = np.linalg.solve(G, y)
    a = np.array([[A[0], A[1]], [A[2], A[3]]])
    b = np.array([A[4], A[5]])

    return A, a, b


@jit('(f8[:,:], f8[:])', nopython=True)
def get_optimum_value(a, b):
    #ans1 = np.linalg.solve(a, -b)
    #return ans1

    epsilon = 1.0E-7 # for avoiding 0 division.
    det = a[0,0]*a[1,1] - a[0,1]*a[1,0]
    ans2 = ((-a[1,1]*b[0] + a[0,1]*b[1])/(det + epsilon), 
            (a[1,0]*b[0]-a[0,0]*b[1])/(det + epsilon))
    #print("Optimum value of A6: ", ans1, ans2)
    return ans2


def calc_fit_data(a, b, x, y):
    def __calc_fit_data(x, y):
        """
        This helper function is defined due to make "a" and "b" global. 
        The "numpy.frompyfunc" does not accept multi dimensional array as an
        argument. Therefore, first of all, I define a helper function that
        accept multi dimensional array as an argument, then pass that helper
        function to numpy.frompyfunc to make that function universal one.
        
        NOTE: np.frompyfunc is NOT a function compatible with numba.
        """
        result = a @ np.array([x, y]).reshape(2, 1) + b.reshape(2, 1)
        return np.double(result[0]), np.double(result[1])

    __calc = np.frompyfunc(__calc_fit_data, 2, 2)

    return __calc(x, y)


def plot(df, a, b, xmin=-25, xmax=25, ymin=-25, ymax=25):
    length = 2
    X = df.a6x.unique()
    Y = df.a6y.unique()
    fig, ax = plt.subplots()

    df.plot(x='real', y='imag', 
            ls='', marker='o', alpha=0.8, ax=ax)
            #ls='', marker='o', color='royalblue', alpha=0.8, ax=ax)

    for xi in X:
        xt = np.zeros(length)
        xt[:] = xi
        yt = np.linspace(-30, 30, length)
        xx, yy = calc_fit_data(a, b, xt, yt)
        ax.plot(xx, yy, marker='', ls='-.',  lw=1, alpha=1.0, label='TTc')
        #ax.plot(xx, yy, marker='', ls='-.',  lw=1, alpha=1.0, color='darkorange')

    for yi in Y:
        yt = np.zeros(length)
        yt[:] = yi
        xt = np.linspace(-30, 30, length)
        xx, yy = calc_fit_data(a, b, xt, yt)
        ax.plot(xx, yy, marker='', ls='-.', lw=1, alpha=1.0)
        #ax.plot(xx, yy, marker='', ls='-.', lw=1, alpha=1.0, color='darkorange')

    optimumvalue = get_optimum_value(a, b)
    #ax.plot(optimumvalue[0], optimumvalue[1], marker='x', ms=5, color='gold')
    ax.plot(0, 0, marker='x', ms=15, color='red') 
    ax.annotate("(%.2f, %.2f)" % (optimumvalue[0], optimumvalue[1]), xy=(1, 1), size=15)
    ax.axis('scaled')
    ax.grid(ls='dashed', color='gray', alpha=0.5)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.savefig('a6_optimum.png', dpi=150, bbox_inches='tight')

    #plt.show()

