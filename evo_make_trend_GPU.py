#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path


def process_fitness_trends(population_size:int,
                           chromosome_size:int,
                           tgt_time="20240427-141320",
                           num_generations=512,
                           num_experiments=10,):

    # ファイルパターンにマッチするファイルのPathオブジェクトを生成する
    names = ["generation", "mean", "min", "max", "std"]

    path = Path(".")
    pattern = f"fitnesstrend_{tgt_time}_{population_size}_{chromosome_size}_*.csv"
    files = path.glob(pattern)

    if not files:
        str1 = f"No files found for population size {population_size}"
        str2 = f"and chromosome size {chromosome_size}"
        print(str1, str2)
        return
    else:
        str1 = f"Processing files for population size {population_size}"
        str2 = f"and chromosome size {chromosome_size}"
        print(str1, str2)

    # 統計情報を格納するためのrow方向が世代、column方向が実験回数の行列を作成する
    min_values = np.zeros((num_generations, num_experiments))
    max_values = np.zeros((num_generations, num_experiments))
    mean_values = np.zeros((num_generations, num_experiments))

    # 同じ測定条件のファイルを開き、列ごとに統計情報を計算する
    for index, file_path in enumerate(files):
        #print(f"Processing file: {file_path}")
        data = pd.read_csv(file_path, names=names)
        min_values[:, index] = data.loc[:, "min"]
        max_values[:, index] = data.loc[:, "max"]
        mean_values[:, index] = data.loc[:, "mean"]

    min_avg = min_values.mean(axis=1)
    max_avg = max_values.mean(axis=1)
    mean_avg = mean_values.mean(axis=1)

    # 結果を新しいDataFrameに保存する
    results = pd.DataFrame({
        "generation": range(1, len(min_avg) + 1),
        "min": min_avg,
        "max": max_avg,
        "mean": mean_avg
    })

    # 結果をCSVファイルに保存する
    output_file = f"fitnesstrend_{tgt_time}_{population_size}_{chromosome_size}_avg.csv"
    results.to_csv(output_file, index=False)


def process_all_combinations():
    for pop_size in range(128, 1024+1, 128):
        for chrom_size in range(128, 1024+1, 128):
            process_fitness_trends(pop_size, chrom_size)


