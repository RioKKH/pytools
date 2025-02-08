#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --- 合成データの作成 ---
np.random.seed(0)  # 再現性のため

# 周波数軸の作成（例：0～1000 Hz, 1000点）
freq = np.linspace(0, 1000, 1000)


# ガウシアンピーク関数
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))


# リファレンス（参照）スペクトル：ピーク位置 [200, 400, 700] Hz
ref = (
    gaussian(freq, 1.0, 200, 10)
    + gaussian(freq, 0.8, 400, 15)
    + gaussian(freq, 1.2, 700, 10)
)
ref += 0.1 * np.sin(0.01 * freq)  # 緩やかな背景
ref += np.random.normal(0, 0.05, freq.shape)  # ノイズの追加

# ターゲットスペクトル：ピーク位置が若干ずれて [205, 395, 705] Hz
tgt = (
    gaussian(freq, 1.0, 205, 10)
    + gaussian(freq, 0.85, 395, 15)
    + gaussian(freq, 1.1, 705, 10)
)
tgt += 0.1 * np.sin(0.01 * freq + 0.1)  # 背景（若干位相ずれ）
tgt += np.random.normal(0, 0.05, freq.shape)  # ノイズの追加

# --- 前処理：平滑化と背景除去 ---
# 1. 平滑化（FFT後のスペクトルとみなす）→ガウシアンフィルタを適用
ref_smooth = gaussian_filter1d(ref, sigma=2)
tgt_smooth = gaussian_filter1d(tgt, sigma=2)

# 2. 背景除去（ベースライン補正）
# ここでは、より大きなスケールのガウシアンフィルタを用いて背景を推定し引いています
ref_corrected = ref_smooth - gaussian_filter1d(ref_smooth, sigma=50)
tgt_corrected = tgt_smooth - gaussian_filter1d(tgt_smooth, sigma=50)

# --- ピーク検出 ---
# 閾値（height）や最小ピーク間隔（distance）を設定してピーク検出
peaks_ref, props_ref = find_peaks(ref_corrected, height=0.1, distance=20)
peaks_tgt, props_tgt = find_peaks(tgt_corrected, height=0.1, distance=20)

# ピークの周波数位置と強度を取得
ref_peaks_freq = freq[peaks_ref]
ref_peaks_amp = props_ref["peak_heights"]
tgt_peaks_freq = freq[peaks_tgt]
tgt_peaks_amp = props_tgt["peak_heights"]

# ピーク情報を (frequency, amplitude) の2次元ポイントとしてまとめる
ref_points = np.column_stack((ref_peaks_freq, ref_peaks_amp))
tgt_points = np.column_stack((tgt_peaks_freq, tgt_peaks_amp))

print("Reference peaks (frequency, amplitude):")
print(ref_points)
print("Target peaks (frequency, amplitude):")
print(tgt_points)


# --- DTW の実装 ---
def euclidean(a, b):
    """2点間のユークリッド距離"""
    return np.linalg.norm(a - b)


def dtw(x, y, dist_func):
    """
    シンプルな DTW 実装
    x: 配列（長さ n） of d次元点
    y: 配列（長さ m） of d次元点
    dist_func: 2点間の距離を計算する関数
    戻り値：最小 DTW 距離, DTW 経路（インデックスのタプルのリスト）
    """
    n, m = len(x), len(y)
    # (n+1) x (m+1) のコスト行列（初期値は無限大）
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # 動的計画法によるコスト計算
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_func(x[i - 1], y[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # 上
                dtw_matrix[i, j - 1],  # 左
                dtw_matrix[i - 1, j - 1],
            )  # 対角線

    # 経路の復元（バックトラッキング）
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        # 三方向のコストを比較して最小の方向へ移動
        diag = dtw_matrix[i - 1, j - 1]
        up = dtw_matrix[i - 1, j]
        left = dtw_matrix[i, j - 1]
        if diag <= up and diag <= left:
            i, j = i - 1, j - 1
        elif up < left:
            i = i - 1
        else:
            j = j - 1
    path.reverse()

    return dtw_matrix[n, m], path


# DTW を実行
dtw_distance, warping_path = dtw(ref_points, tgt_points, euclidean)
print("\nDTW distance:", dtw_distance)
print("Warping path (ref_index, tgt_index):", warping_path)

# --- 結果の可視化 ---
plt.figure(figsize=(12, 8))

# スペクトルのプロット
plt.subplot(2, 1, 1)
plt.plot(freq, ref_corrected, label="Reference (corrected)", color="blue")
plt.plot(freq, tgt_corrected, label="Target (corrected)", color="orange")
plt.scatter(
    ref_peaks_freq, ref_peaks_amp, color="blue", marker="o", s=100, label="Ref Peaks"
)
plt.scatter(
    tgt_peaks_freq, tgt_peaks_amp, color="orange", marker="x", s=100, label="Tgt Peaks"
)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (after correction)")
plt.title("Smoothed & Baseline-Corrected Spectra with Detected Peaks")
plt.legend()

# DTW 経路の可視化
plt.subplot(2, 1, 2)
plt.imshow(
    np.zeros((len(ref_points), len(tgt_points))), cmap="gray", interpolation="nearest"
)
# DTW 経路をプロット（各点は ref のピークと tgt のピークの対応を示す）
for i, j in warping_path:
    plt.plot(j, i, "r.-", markersize=15)
plt.xlabel("Target peak index")
plt.ylabel("Reference peak index")
plt.title("DTW Warping Path between Peak Sequences")
plt.gca().invert_yaxis()  # 上に小さいインデックスが来るように
plt.tight_layout()
plt.show()
