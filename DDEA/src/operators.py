#!/usr/bin/env python

import numpy as np


def SBX(POP, bu, bd, pc, n):
    """
    Simulated Binary Crossover (SBX) による交叉操作

    Parameters:
      POP : numpy.ndarray
          現在の個体群 (形状: (pop_size, num_variables + 付加情報))。最初の列は決定変数。
      bu : numpy.ndarray
          各決定変数の上限 (形状: (num_variables,))
      bd : numpy.ndarray
          各決定変数の下限 (形状: (num_variables,))
      pc : float
          交叉確率
      n : int
          交叉を行う母集団サイズ

    Returns:
      NPOP : numpy.ndarray
          交叉により生成された子個体群 (形状: (2*n, num_variables))
    """
    eta_c = 15
    pop_size, _ = POP.shape
    num_variables = len(bu)
    NPOP = np.zeros((2 * n, num_variables))

    for i in range(n):
        r1 = np.random.rand()
        if r1 <= pc:
            indices = np.random.permutation(pop_size)
            parent1 = POP[indices[0], :num_variables]
            parent2 = POP[indices[1], :num_variables]
            child1 = np.zeros(num_variables)
            child2 = np.zeros(num_variables)
            for j in range(num_variables):
                par1 = parent1[j]
                par2 = parent2[j]
                yd = bd[j]
                yu = bu[j]
                r2 = np.random.rand()
                if r2 <= 0.5:
                    y1 = min(par1, par2)
                    y2 = max(par1, par2)
                    if abs(y2 - y1) < 1e-14:
                        child1[j] = par1
                        child2[j] = par2
                    else:
                        if (y1 - yd) > (yu - y2):
                            beta = 1 + 2 * (yu - y2) / (y2 - y1)
                        else:
                            beta = 1 + 2 * (y1 - yd) / (y2 - y1)
                        beta = 1.0 / beta
                        expp = eta_c + 1.0
                        alpha = 2.0 - beta**expp
                        r3 = np.random.rand()
                        if r3 <= 1.0 / alpha:
                            alpha_val = alpha * r3
                            expp_val = 1.0 / (eta_c + 1.0)
                            betaq = alpha_val**expp_val
                        else:
                            alpha_val = 1.0 / (2.0 - alpha * r3)
                            expp_val = 1.0 / (eta_c + 1.0)
                            betaq = alpha_val**expp_val
                        child1[j] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                        child2[j] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                        child1[j] = min(max(child1[j], yd), yu)
                        child2[j] = min(max(child2[j], yd), yu)
                else:
                    child1[j] = parent1[j]
                    child2[j] = parent2[j]
            NPOP[2 * i, :] = child1
            NPOP[2 * i + 1, :] = child2
        else:
            individual = POP[i, :num_variables]
            NPOP[2 * i, :] = individual
            NPOP[2 * i + 1, :] = individual
    return NPOP


def mutation(POP, bu, bd, pm, n):
    """
    多項式突然変異による変異操作

    Parameters:
      POP : numpy.ndarray
          現在の個体群 (形状: (pop_size, num_variables + 付加情報))。最初の列は決定変数。
      bu : numpy.ndarray
          各決定変数の上限 (形状: (num_variables,))
      bd : numpy.ndarray
          各決定変数の下限 (形状: (num_variables,))
      pm : float
          突然変異確率
      n : int
          変異を行う個体数

    Returns:
      NPOP : numpy.ndarray
          変異後の個体群 (形状: (n, num_variables))
    """
    eta_m = 15
    pop_size, _ = POP.shape
    num_variables = len(bu)
    NPOP = POP[:n, :num_variables].copy()

    for i in range(n):
        for j in range(num_variables):
            if np.random.rand() <= pm:
                y = NPOP[i, j]
                yd = bd[j]
                yu = bu[j]
                if y > yd:
                    if (y - yd) < (yu - y):
                        delta = (y - yd) / (yu - yd)
                    else:
                        delta = (yu - y) / (yu - yd)
                    r2 = np.random.rand()
                    indi = 1.0 / (eta_m + 1.0)
                    if r2 <= 0.5:
                        xy = 1.0 - delta
                        val = 2 * r2 + (1 - 2 * r2) * (xy ** (eta_m + 1))
                        deltaq = (val**indi) - 1.0
                    else:
                        xy = 1.0 - delta
                        val = 2 * (1 - r2) + 2 * (r2 - 0.5) * (xy ** (eta_m + 1))
                        deltaq = 1.0 - (val**indi)
                    y = y + deltaq * (yu - yd)
                    NPOP[i, j] = min(max(y, yd), yu)
                else:
                    NPOP[i, j] = np.random.rand() * (yu - yd) + yd
    return NPOP
