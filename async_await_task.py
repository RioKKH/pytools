#!/usr/bin/env python

import asyncio
import time


async def function_1(sec):
    print(f"{sec}秒待ちます")
    await asyncio.sleep(sec)
    return f"{sec}秒の待機に成功しました"


async def main():
    """
    並行処理される。２秒で処理される
    """
    print(f"main開始 {time.strftime('%X')}")
    task1 = asyncio.create_task(function_1(1))
    task2 = asyncio.create_task(function_1(2))
    # task1 = asyncio.create_task(asyncio.sleep(1))
    # task2 = asyncio.create_task(asyncio.sleep(2))
    await task1
    await task2
    print(f"{task1.result()}")
    print(f"{task2.result()}")
    print(f"main修了 {time.strftime('%X')}")


if __name__ == "__main__":
    asyncio.run(main())
