#!/usr/bin/env python

import asyncio
import time


async def function_1(sec):
    print(f"{sec}秒待ちます")
    await asyncio.sleep(sec)
    return f"{sec}秒の待機に成功しました"


async def main():
    print(f"main開始 {time.strftime('%X')}")
    try:
        # タイムアウトの指定ができる
        result = await asyncio.wait_for(function_1(10), timeout=3)
        print(result)
    except asyncio.TimeoutError:
        print("タイムアウト")
    print(f"main終了 {time.strftime('%X')}")


if __name__ == "__main__":
    asyncio.run(main())
