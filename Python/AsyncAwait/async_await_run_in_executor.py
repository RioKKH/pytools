#!/usr/bin/env python

import asyncio
import time


async def function_1(sec):
    print(f"{sec}秒待ちます")
    loop = asyncio.get_running_loop()
    # 第1引数: Executor
    # 第2引数: 関数
    # 第2引数: 関数の引数
    await loop.run_in_executor(None, time.sleep, sec)
    return f"{sec}秒の待機に成功しました"


async def main():
    print(f"main開始 {time.strftime('%X')}")
    results = await asyncio.gather(function_1(1), function_1(2))
    print(results)
    print(f"main修了 {time.strftime('%X')}")


if __name__ == "__main__":
    asyncio.run(main())
