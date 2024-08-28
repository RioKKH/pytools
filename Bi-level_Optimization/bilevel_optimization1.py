#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize

# ベンチマーク問題: 簡単な二次関数の最小化
def upper_objective(x:float, y:float) -> float:
    """上位問題の目的関数(リーダーの問題)

    上位問題の目的関数になります。
    Bi-level最適化では上位問題の目的関数や制約条件が、
    下位問題の最適解に依存します。
    一般に、解きたい「本当の問題」は上位問題になります。
    上位問題は全体的な目標や主要な意思決定を表現する為です。

    Args:
        x (float): xの値
        y (float): yの値

    Returns:
        float: 上位問題の目的関数の値
    """ 
    return (x - 3)**2 + (y - 2)**2

def lower_objective(y:float, x:float) -> float:
    """下位問題の目的関数(フォロワーの問題)

    下位問題の目的関数になります。
    下位問題は上位問題の決定に対する反応や従属的なさいてきかを表します。

    Args:
        y (float): yの値
        x (float): xの値
    """
    return (y - x)**2

def solve_lower_problem(x):
    res = minimize(lower_objective, x0=0, args=(x,), method="BFGS")
    return res.x[0]

def bi_level_objective(x):
    y = solve_lower_problem(x)
    return upper_objective(x, y)

# ベンチマーク問題の解法
x_opt = minimize(bi_level_objective, x0=0, method="BFGS").x[0]
y_opt = solve_lower_problem(x_opt)

print(f"ベンチマーク問題の最適解: x = {x_opt:.4f}, y = {y_opt:.4f}")
print(f"最適値: {upper_objective(x_opt, y_opt):.4f}")
