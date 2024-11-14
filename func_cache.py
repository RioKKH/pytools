#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
from functools import cache
from typing import Callable
from tabulate import tabulate

# キャッシュ有り版
@cache
def fibonacci_cached(i: int) -> int:
    if i == 0 or i == 1:
        return i
    else:
        return fibonacci_cached(i - 1) + fibonacci_cached(i - 2)

# キャッシュ無版
def fibonacci_uncached(i: int) -> int:
    if i == 0 or i == 1:
        return i
    else:
        return fibonacci_uncached(i - 1) + fibonacci_uncached(i - 2)

def measure_execution_time(func: Callable[[int], int], n: int, iterations: int = 3) -> float:
    """
    指定された関数の実行時間を計測する

    Args:
        func: 計測対象の関数
        n: フィボナッチ数列のn番目
        iterations: 計測を繰り返す回数

    Returns:
        float: 平均実行時間(秒)
    """
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        func(n)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    return sum(times) / iterations

def compare_performance(test_numbers: list[int]) -> None:
    """
    キャッシュ有りと無しの実行時間を比較して表示する

    Args:
        test_numbers: テストする数値のリスト
    """
    for n in test_numbers:
        results = []
        cached_time = measure_execution_time(fibonacci_cached, n)
        uncached_time = measure_execution_time(fibonacci_uncached, n)
        speedup = uncached_time / cached_time if cached_time > 0 else float('inf')

        results.append([
            n,
            f"{cached_time:.6f}",
            f"{uncached_time}",
            f"{speedup:.2f}x"
        ])

    # 結果を表形式で表示する
    headers = ["n", "キャッシュ有り(秒)", "キャッシュ無し(秒)", "高層化率"]
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    # テストする数値のリスト
    test_numbers = [10, 20, 30, 35]
    print("フィボナッチ数列の実行時間比較\n")
    compare_performance(test_numbers)


