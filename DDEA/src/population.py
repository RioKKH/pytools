#!/usr/bin/env python

import numpy as np
from .utils import lhs_design


def initialize_pop(n, c, bu, bd):
    """
    ラテンハイパーキューブサンプリングを用いた初期個体群の生成

    Parameters:
      n : int
          個体数（母集団サイズ）
      c : int
          決定変数の数
      bu : numpy.ndarray
          各決定変数の上限 (形状: (c,))
      bd : numpy.ndarray
          各決定変数の下限 (形状: (c,))

    Returns:
      POP : numpy.ndarray
          初期個体群 (形状: (n, c))
    """
    samples = lhs_design(n, c)
    POP = samples * (bu - bd) + bd
    return POP
