#!/usr/bin/env python

import asyncio
import time


async def function_1(sec):
    print(f"{sec}秒待ちます")
    await asyncio.sleep(sec)
    return f"{sec}秒の待機に成功しました"


async def main():
    """
    asyncio.gather: 引数に並行処理したいコルーチンやタスクを指定すればOK
    コルーチンを複数実行し、その戻り値をまとめて受け取る事が出来るもの
    """
    print(f"main開始 {time.strftime('%X')}")
    task1 = asyncio.create_task(function_1(1))
    # コルーチンとタスクを同時に引数として渡している
    results = await asyncio.gather(function_1(2), task1)
    print(results)
    print(f"main修了 {time.strftime('%X')}")


if __name__ == "__main__":
    asyncio.run(main())
