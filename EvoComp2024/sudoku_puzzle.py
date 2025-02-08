#!/usr/bin/env python

import numpy as np
from numpy import ndarray
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize


class SudokuPuzzle:
    """数独パズルの状態を管理し、評価を行うクラス"""

    def __init__(self, hint_pattern: list[int]) -> None:
        """コンストラクター
        Parameters
        -----------
        hint_pattern : list[int]
            ヒントを配置出来る場所を示すパターン
            (0: 配置不可, 1: 配置可能)
        """
        self.hint_pattern = np.array(hint_pattern)
        self.size = 9
        self.block_size = 3

    def is_valid_sudoku(self, board: SudokuBoard) -> bool:
        """数独のルールに従っているかチェックする

        Parameters
        ----------
        board : SudokuBoard
            9x9の数独盤面

        Returns
        -------
        bool
            ルールに従っている場合True
        """
        # 行のチェック
        for i in range(self.size):
            row = board[i]
            if not self._is_valid_unit(row[row != 0]):
                return False

        # 行のチェック
        for j in range(self.size):
            col = board[:, j]
            if not self._is_valid_unit(col[col != 0]):
                return False

        # 3x3ブロックのチェック
        for block_i in range(self.block_size):
            for block_j in range(self.block_size):
                block = board[
                    block_i * 3 : (block_i + 1) * 3, block_j * 3 : (block_j + 1) * 3
                ].flatten()
                if not self._is_valid_unit(block[block != 0]):
                    return False

        return True

    @staticmethod
    def _is_valid_unit(unit: ndarray[tuple[int], np.int_]) -> bool:
        """行、列、ブロックの数字の重複をチェックする"""
        return len(set(unit)) == len(unit)

    def evaluate_tension(self, solution: Solution) -> float:
        
