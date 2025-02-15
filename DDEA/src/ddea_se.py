#!/usr/bin/env python

import numpy as np
import time
from .population import initialize_pop
from .rbf import RBF_EnsembleUN, RBF_Ensemble_predictor
from .operators import SBX, mutation
from .selection import SelectModels


def DDEA_SE(c, L, bu, bd, record_history=False):
    """
    Offline Data-Driven Evolutionary Optimization using Selective Surrogate Ensembles (DDEA_SE)
    * record_history=Trueの時、2次元の場合に各世代の個体群とサロゲート予測値を記録して返す

    Parameters:
        c : int
            決定変数の数。
        L : numpy.ndarray
            オフラインデータ。各行は [c個の決定変数, 正確な目的関数値] の形式。
        bu : numpy.ndarray
            各決定変数の上限（1次元配列）。
        bd : numpy.ndarray
            各決定変数の下限（1次元配列）。
        record_history : bool
            2次元の場合に各世代の状態を記録するか否か

    Returns:
        exec_time : float
            実行時間（秒）。
        P : numpy.ndarray
            最終的な予測最適解（決定変数ベクトル）。
        gbest : numpy.ndarray
            各世代の最良解の履歴（各行が1世代分）。
        pop_history (optional) : list of np.ndarray
            各世代の個体群 (決定変数部分)
        surrogate_history (optional) : list of tuple(X1, X2, Z)
            各世代でのグリッド上のサロゲート予測値 (2次元の場合)
    """
    np.random.seed(int(time.time()))
    # パラメータ設定
    num_neurons = (
        c  # RBFモデルにおける各モデルのニューロン数（MATLAB実装と同様に c を使用）
    )
    T = 2000  # RBFモデルの総数
    Q = 100  # サロゲートモデルの選択数
    gmax = 100  # 最大世代数
    pc = 1.0  # 交叉確率
    pm = 1.0 / c  # 突然変異確率
    pop_size = 100  # 個体群サイズ

    # オフラインデータからRBFモデルのプール（アンサンブル）を構築
    W, B, C_arr, S_arr = RBF_EnsembleUN(L, c, num_neurons, T)
    # 初期は全モデルを使用
    model_indices = np.arange(T)

    # ラテンハイパーキューブサンプリングで初期個体群を生成
    POP = initialize_pop(pop_size, c, bu, bd)  # 形状: (pop_size, c)
    # サロゲートモデルによる目的値予測（各モデルごとの予測値を計算）
    Y = RBF_Ensemble_predictor(
        W[model_indices],
        B[model_indices],
        C_arr[model_indices],
        S_arr[model_indices],
        POP,
        c,
    )
    # 各個体に予測値を付加
    POP = np.hstack((POP, Y))

    g = 1
    gbest = []
    # 以下、履歴記録用のリスト (c==2かつrecord_history=Trueの場合)
    pop_history = [] if record_history and c == 2 else None
    surrogate_history = [] if record_history and c == 2 else None

    start_time = time.time()

    # 2次元用のグリッド作成 (後のサロゲート予測用)
    if c == 2 and record_history:
        x1_lin = np.linspace(bd[0], bu[0], 10)
        x2_lin = np.linspace(bd[1], bu[1], 10)
        # x1_lin = np.linspace(bd[0], bu[0], 100)
        # x2_lin = np.linspace(bd[1], bu[1], 100)
        X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
        grid_points = np.c_[X1_grid.ravel(), X2_grid.ravel()]

    while g <= gmax:
        # モデル管理：最良個体に基づきサロゲートモデルのサブセットを選択
        if g != 1:
            model_indices = SelectModels(
                W, B, C_arr, S_arr, POP[0, :c].reshape(1, -1), c, Q
            )
            # 個体は決定変数部分のみを残す
            POP = POP[:, :c]
            # 選択したモデルで予測値を再計算し、個体に付加
            Y = RBF_Ensemble_predictor(
                W[model_indices],
                B[model_indices],
                C_arr[model_indices],
                S_arr[model_indices],
                POP,
                c,
            )
            POP = np.hstack((POP, Y))

        # 記録：原個体群(決定変数部分)を保存(２次元の場合)
        if record_history and c == 2:
            pop_history.append(POP[:, :c].copy())
            # サロゲート予測の計算
            Y_grid = RBF_Ensemble_predictor(
                W[model_indices],
                B[model_indices],
                C_arr[model_indices],
                S_arr[model_indices],
                grid_points,
                c,
            )
            # 平均予測値
            Z = np.mean(Y_grid, axis=1).reshape(X1_grid.shape)
            surrogate_history.append((X1_grid, X2_grid, Z))

        # 交叉（SBX）による子個体生成
        NPOP1 = SBX(POP, bu, bd, pc, pop_size)
        Y1 = RBF_Ensemble_predictor(
            W[model_indices],
            B[model_indices],
            C_arr[model_indices],
            S_arr[model_indices],
            NPOP1,
            c,
        )
        NPOP1 = np.hstack((NPOP1, Y1))

        # 突然変異による子個体生成
        NPOP2 = mutation(POP, bu, bd, pm, pop_size)
        Y2 = RBF_Ensemble_predictor(
            W[model_indices],
            B[model_indices],
            C_arr[model_indices],
            S_arr[model_indices],
            NPOP2,
            c,
        )
        NPOP2 = np.hstack((NPOP2, Y2))

        # 現在の個体群と子個体群を結合
        POP = np.vstack((POP, NPOP1, NPOP2))
        # 各個体の予測値部分の平均で評価
        YAVE = np.mean(POP[:, c:], axis=1)
        # 評価値が小さい上位pop_size個体を選択
        sorted_indices = np.argsort(YAVE)
        POP = POP[sorted_indices[:pop_size], :c]

        # 現世代の最良解を記録
        P = POP[0, :]
        gbest.append(P.copy())
        g += 1

    exec_time = time.time() - start_time
    gbest = np.array(gbest)

    if record_history and c == 2:
        return exec_time, P, gbest, pop_history, surrogate_history
    else:
        return exec_time, P, gbest


if __name__ == "__main__":
    # 使用例：決定変数5個、オフラインデータ50サンプルの簡単な問題設定
    c = 5
    num_samples = 50
    bd = np.zeros(c)
    bu = np.ones(c) * 10
    # オフラインデータ：ランダムに生成した決定変数と、例として二乗和の目的関数値
    X = np.random.uniform(bd, bu, size=(num_samples, c))
    y = np.sum(X**2, axis=1)
    L = np.hstack((X, y.reshape(-1, 1)))

    exec_time, P, gbest = DDEA_SE(c, L, bu, bd)
    print("Execution Time:", exec_time)
    print("Final Best Solution:", P)
    print("History of Best Solutions:\n", gbest)
