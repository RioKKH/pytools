#!/usr/bin/env python


"""
逐次処理
前の処理が修了したら次の処理が実行される

非同期処理
処理の完了を待たずに別の処理を実行できる
処理が完了するまでに時間が掛かるときに使う
ファイルのI/Oとか
ファイルのアップロード・ダウンロード
データベースの読み書き

並行処理を簡単に扱えるようにしたものが
async/awaitの非同期処理
イベントループがタスクを良い感じに制御してくれる
"""

import asyncio
import time


async def function_1(sec):
    print(f"{sec}秒待ちます")
    await asyncio.sleep(sec)
    return f"{sec}秒の待機に成功しました"


async def main():
    """
    asyncがついている→コルーチンと呼ぶ
    処理をある場所で一時中段・再開出来るもの
    このコードはシーケンシャルに実行されるので、
    ３秒かかることに注意。つまりまだ平行処理には
    なっていない
    """

    print(f"main開始 {time.strftime('%X')}")
    # asyncio.sleep()はコルーチン
    # コルーチンが完了するまで待つにはawaitを付ける必要がある。
    result_1 = await function_1(1)
    result_2 = await function_1(2)
    print(result_1)
    print(result_2)
    print(f"main 修了 {time.strftime('%X')}")


if __name__ == "__main__":
    asyncio.run(main())
