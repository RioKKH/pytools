#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --------------------------
# 1. 合成データの作成（電気的ノイズをシミュレーション）
# --------------------------
np.random.seed(42)  # 再現性のため

# 周波数軸：0～1000 Hz, 1000点
freq = np.linspace(0, 1000, 1000)

# ノイズ（背景）の生成：平均0、標準偏差0.1
ref_noise = np.random.normal(0, 0.1, 1000)
tgt_noise = np.random.normal(0, 0.1, 1000)

# リファレンスとターゲットのスペクトル
ref = ref_noise.copy()
tgt = tgt_noise.copy()

# 急峻なスパイクを追加
# リファレンス：250 Hz と 700 Hz にスパイク（単一点）
ref_peak_indices = [250, 700]
for idx in ref_peak_indices:
    ref[idx] += 5.0

# ターゲット：ピーク位置が若干ずれて 253 Hz と 697 Hz
tgt_peak_indices = [253, 697]
for idx in tgt_peak_indices:
    tgt[idx] += 5.0

# --------------------------
# 2. 前処理（平滑化＆ベースライン補正）
# --------------------------
# 2-1. 平滑化（ノイズを多少低減；ここでは sigma=1 で実施）
ref_smooth = gaussian_filter1d(ref, sigma=1)
tgt_smooth = gaussian_filter1d(tgt, sigma=1)

# 2-2. 背景除去（大きなスケールのガウシアンフィルタで背景推定）
ref_corrected = ref_smooth - gaussian_filter1d(ref_smooth, sigma=50)
tgt_corrected = tgt_smooth - gaussian_filter1d(tgt_smooth, sigma=50)

# --------------------------
# 3. ピーク検出
# --------------------------
# find_peaks でピーク検出（height=0.5, distance=5）
peaks_ref, props_ref = find_peaks(ref_corrected, height=0.5, distance=5)
peaks_tgt, props_tgt = find_peaks(tgt_corrected, height=0.5, distance=5)

# 検出されたピークの周波数位置と高さ
ref_peaks_freq = freq[peaks_ref]
ref_peaks_amp = props_ref["peak_heights"]
tgt_peaks_freq = freq[peaks_tgt]
tgt_peaks_amp = props_tgt["peak_heights"]

# ピーク情報を (frequency, amplitude) の2次元データにまとめる
ref_points = np.column_stack((ref_peaks_freq, ref_peaks_amp))
tgt_points = np.column_stack((tgt_peaks_freq, tgt_peaks_amp))

print("Reference peaks (frequency, amplitude):")
print(ref_points)
print("Target peaks (frequency, amplitude):")
print(tgt_points)


# --------------------------
# 4. シンプルな DTW の実装と適用
# --------------------------
def euclidean(a, b):
    """2点間のユークリッド距離"""
    return np.linalg.norm(a - b)


def dtw(x, y, dist_func):
    """
    シンプルな DTW 実装
    x: 配列（長さ n） of d次元点
    y: 配列（長さ m） of d次元点
    dist_func: 2点間の距離計算関数
    戻り値：最小DTW距離, 経路（インデックスのタプルのリスト）
    """
    n, m = len(x), len(y)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_func(x[i - 1], y[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # 上
                dtw_matrix[i, j - 1],  # 左
                dtw_matrix[i - 1, j - 1],
            )  # 対角線

    # バックトラッキングで経路復元
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
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


# DTW の実行
dtw_distance, warping_path = dtw(ref_points, tgt_points, euclidean)
print("\nDTW distance:", dtw_distance)
print("Warping path (ref_index, tgt_index):", warping_path)

# --------------------------
# 5. プロット（前処理の様子と DTW 経路）
# --------------------------
plt.figure(figsize=(12, 10))

# (A) スペクトルのプロット
plt.subplot(2, 1, 1)
# オリジナルのノイジーなデータ（半透明：alpha=0.3）をプロット
plt.plot(freq, ref, color="blue", linestyle=":", alpha=0.3, label="Ref Original")
plt.plot(freq, tgt, color="orange", linestyle=":", alpha=0.3, label="Tgt Original")
# 平滑化＆ベースライン補正後のデータ
plt.plot(freq, ref_corrected, color="blue", label="Ref Processed")
plt.plot(freq, tgt_corrected, color="orange", label="Tgt Processed")
# 検出されたピークをプロット
plt.scatter(
    ref_peaks_freq,
    ref_peaks_amp,
    color="blue",
    marker="o",
    s=100,
    zorder=3,
    label="Ref Peaks",
)
plt.scatter(
    tgt_peaks_freq,
    tgt_peaks_amp,
    color="orange",
    marker="x",
    s=100,
    zorder=3,
    label="Tgt Peaks",
)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (after correction)")
plt.title("Noisy Data and Processed Spectrum with Detected Peaks")
plt.legend()
plt.grid(True)

# (B) DTW 経路の可視化
plt.subplot(2, 1, 2)
# 背景を白くするため、全体を 1 の値で埋めた画像を表示
dtw_bg = np.ones((len(ref_points), len(tgt_points)))
plt.imshow(
    dtw_bg,
    cmap="gray",
    interpolation="nearest",
    extent=[-0.5, len(tgt_points) - 0.5, len(ref_points) - 0.5, -0.5],
)
plt.colorbar(label="Background (white)")
# 経路を赤い線と大きめのマーカーでプロット
path_ref_idx = [i for i, j in warping_path]
path_tgt_idx = [j for i, j in warping_path]
plt.plot(
    path_tgt_idx, path_ref_idx, "r.-", linewidth=2, markersize=12, label="DTW Path"
)
plt.xlabel("Target Peak Index")
plt.ylabel("Reference Peak Index")
plt.title("DTW Warping Path between Peak Sequences")
plt.legend()
plt.gca().invert_yaxis()  # Y軸の順序を反転して上に小さいインデックスが来るように
plt.grid(True)

plt.tight_layout()
plt.show()
