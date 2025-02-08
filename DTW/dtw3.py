#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from collections import defaultdict

# ================================
# 1. データ生成（電気的ノイズのシミュレーション）
# ================================
np.random.seed(42)  # 再現性のため

# 周波数軸：0～1000 Hz, 1000点
freq = np.linspace(0, 1000, 1000)

# 背景ノイズ：平均0、標準偏差0.1
ref_noise = np.random.normal(0, 0.1, 1000)
tgt_noise = np.random.normal(0, 0.1, 1000)

# リファレンスとターゲットの元データ
ref = ref_noise.copy()
tgt = tgt_noise.copy()

# 急峻なスパイクを追加
# リファレンス：250 Hz と 700 Hz にスパイク（＋5.0）
ref_peak_indices = [250, 700]
for idx in ref_peak_indices:
    ref[idx] += 5.0

# ターゲット：ピーク位置が若干ずれて 253 Hz と 697 Hz にスパイク（＋5.0）
tgt_peak_indices = [253, 697]
for idx in tgt_peak_indices:
    tgt[idx] += 5.0

# ================================
# 2. 前処理：平滑化＆背景除去（ベースライン補正）
# ================================
# 平滑化（sigma=1）
ref_smooth = gaussian_filter1d(ref, sigma=1)
tgt_smooth = gaussian_filter1d(tgt, sigma=1)

# 背景除去：大きなスケール（sigma=50）のガウシアンフィルタで背景推定して引く
ref_corrected = ref_smooth - gaussian_filter1d(ref_smooth, sigma=50)
tgt_corrected = tgt_smooth - gaussian_filter1d(tgt_smooth, sigma=50)

# ================================
# 3. ピーク検出（find_peaks を用いる）
# ================================
# 高さが0.5以上、最小間隔が5点以上のピークを抽出
peaks_ref, props_ref = find_peaks(ref_corrected, height=0.5, distance=5)
peaks_tgt, props_tgt = find_peaks(tgt_corrected, height=0.5, distance=5)

# ピークの周波数位置と振幅（検出されたピークのみ）
ref_peaks_freq = freq[peaks_ref]
ref_peaks_amp = props_ref["peak_heights"]
tgt_peaks_freq = freq[peaks_tgt]
tgt_peaks_amp = props_tgt["peak_heights"]

# 各ピークを (frequency, amplitude) のペアにまとめる
ref_points = np.column_stack((ref_peaks_freq, ref_peaks_amp))
tgt_points = np.column_stack((tgt_peaks_freq, tgt_peaks_amp))

print("Reference peaks (frequency, amplitude):")
print(ref_points)
print("Target peaks (frequency, amplitude):")
print(tgt_points)


# ================================
# 4. DTW によるピーク対応付け
# ================================
def euclidean(a, b):
    """2点間のユークリッド距離"""
    return np.linalg.norm(a - b)


def dtw(x, y, dist_func):
    """
    シンプルな DTW の実装
    x: 配列（長さ n） of d次元点
    y: 配列（長さ m） of d次元点
    dist_func: 2点間の距離計算関数
    戻り値: 最小 DTW 距離, 経路（(ref_index, tgt_index) のタプルのリスト）
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


dtw_distance, warping_path = dtw(ref_points, tgt_points, euclidean)
print("\nDTW distance:", dtw_distance)
print("Warping path (ref_index, tgt_index):", warping_path)

# ================================
# 5. DTW 対応結果を用いた補正処理
# ================================
# warping_path の各ペア (ref_idx, tgt_idx) をもとに、
# 各ターゲットピークに対応するリファレンス側のピーク振幅をグループ化
target_to_ref_amp = defaultdict(list)
for ref_idx, tgt_idx in warping_path:
    target_to_ref_amp[tgt_idx].append(ref_peaks_amp[ref_idx])

# ターゲット側のピーク振幅が一定以上（threshold=0.5）のものについて補正：
# 補正値 = ターゲットピーク振幅 - (対応するリファレンスピーク振幅の平均)
amplitude_threshold = 0.5
corrected_tgt_peaks = {}  # {tgt_peak_index: corrected_amplitude}
for idx, amp in enumerate(tgt_peaks_amp):
    if amp >= amplitude_threshold:
        if idx in target_to_ref_amp:
            ref_amp_avg = np.mean(target_to_ref_amp[idx])
            corrected_amp = amp - ref_amp_avg
        else:
            # 対応がなければ補正は行わず元の値とする
            corrected_amp = amp
        corrected_tgt_peaks[idx] = corrected_amp

print("\nCorrected target peak amplitudes (for peaks above threshold):")
for idx, corr_amp in corrected_tgt_peaks.items():
    print(
        f"Target peak index {idx}: Original amplitude = {tgt_peaks_amp[idx]:.3f}, Corrected amplitude = {corr_amp:.3f}"
    )

# ================================
# 6. 結果のプロット
# ================================
plt.figure(figsize=(12, 12))

# (A) スペクトルと検出されたピークの表示
plt.subplot(3, 1, 1)
# オリジナルのノイジーデータ（半透明：alpha=0.3）
plt.plot(freq, ref, color="blue", linestyle=":", alpha=0.3, label="Ref Original")
plt.plot(freq, tgt, color="orange", linestyle=":", alpha=0.3, label="Tgt Original")
# 平滑化＆ベースライン補正後のデータ
plt.plot(freq, ref_corrected, color="blue", label="Ref Processed")
plt.plot(freq, tgt_corrected, color="orange", label="Tgt Processed")
# 検出されたピーク
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
plt.title("Spectrum and Detected Peaks")
plt.legend()
plt.grid(True)

# (B) DTW 経路の表示
plt.subplot(3, 1, 2)
# 背景を白（値1）で表示
dtw_bg = np.ones((len(ref_points), len(tgt_points)))
plt.imshow(
    dtw_bg,
    cmap="gray",
    interpolation="nearest",
    extent=[-0.5, len(tgt_points) - 0.5, len(ref_points) - 0.5, -0.5],
)
plt.colorbar(label="Background (white)")
# DTW 経路を赤い線と大きめのマーカーで描画
path_ref_idx = [i for i, j in warping_path]
path_tgt_idx = [j for i, j in warping_path]
plt.plot(
    path_tgt_idx, path_ref_idx, "r.-", linewidth=2, markersize=12, label="DTW Path"
)
plt.xlabel("Target Peak Index")
plt.ylabel("Reference Peak Index")
plt.title("DTW Warping Path between Peak Sequences")
plt.legend()
plt.gca().invert_yaxis()  # Y軸を反転して上に小さいインデックスが来るように
plt.grid(True)

# (C) ターゲットピークの補正前後の振幅表示
plt.subplot(3, 1, 3)
# 補正前のターゲットピーク（オレンジの丸印）
plt.plot(
    tgt_peaks_freq,
    tgt_peaks_amp,
    "o",
    color="orange",
    label="Original Tgt Peaks",
    markersize=8,
)
# 補正後のピーク：同じ周波数で補正値（振幅）をプロット（緑の四角印）
corrected_freq = [tgt_peaks_freq[idx] for idx in corrected_tgt_peaks.keys()]
corrected_amp_values = [val for val in corrected_tgt_peaks.values()]
plt.plot(
    corrected_freq,
    corrected_amp_values,
    "s",
    color="green",
    label="Corrected Tgt Peaks",
    markersize=10,
)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Peak Amplitude")
plt.title("Target Peak Amplitudes Before and After Correction")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

