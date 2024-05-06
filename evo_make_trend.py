#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path


def process_fitness_trends(population_size:int,
                           chromosome_size:int,
                           tgt_time="20240427-141320"):
    # ファイルパターンにマッチするファイルのPathオブジェクトを生成する
    path = Path(".")
    pattern = f"fitnesstrend_{tgt_time}_{population_size}_{chromosome_size}_*.csv"
    files = path.glob(pattern)

    #for file_path in files:
    #    print(file_path)

    # if not files:
    #     str1 = f"No files found for population size {population_size}"
    #     str2 = f"and chromosome size {chromosome_size}"
    #     print(str1, str2)
    #     return
    # else:
    #     str1 = f"Processing files for population size {population_size}"
    #     str2 = f"and chromosome size {chromosome_size}"
    #     print(str1, str2)

    # # 各ファイルからデータを読み込み、統計情報を計算する
    min_values = []
    max_values = []
    mean_values = []

    # 同じ測定条件のファイルを開き、列ごとに統計情報を計算する
    for file_path in files:
        print(f"Processing file: {file_path}")
        data = pd.read_csv(file_path)
        min_values.append(data.iloc[:, 2]) # 最小値の列
        max_values.append(data.iloc[:, 3]) # 最大値の列
        mean_values.append(data.iloc[:, 1]) # 平均値の列


    print(min_values)

    # # 列ごとに統計情報を平均する
    # min_df = pd.concat(min_values, axis=1)
    # max_df = pd.concat(max_values, axis=1)
    # mean_df = pd.concat(mean_values, axis=1)

    # min_avg = min_df.mean(axis=1)
    # max_avg = max_df.mean(axis=1)
    # mean_avg = mean_df.mean(axis=1)

    # # 結果を新しいDataFrameに保存する
    # results = pd.DataFrame({
    #     "generation": range(1, len(min_avg) + 1),
    #     "min": min_avg,
    #     "max": max_avg,
    #     "mean": mean_avg
    # })

    # # 結果をCSVファイルに保存する
    # output_file = f"fitnesstrend_{tgt_time}_{population_size}_{chromosome_size}_avg.csv"
    # results.to_csv(output_file, index=False)


def process_all_combinations():
    for pop_size in range(32, 1024+1, 32):
        for chrom_size in range(32, 1024+1, 32):
            process_fitness_trends(pop_size, chrom_size)


