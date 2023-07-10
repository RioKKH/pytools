#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pythonは他の一部のオブジェクト指向言語とは異なり、
メソッドのオーバーロードを直接サポートしていない。しかし、Pythonでは
'@overload'でこれーたを使用して、型ヒントと一緒に関数の異なるバージョンを
定義することが出来る。
この'@overload'でこれーたは主に静的型チェッカー、linter、IDEなどのツールに
よって使用される。

しかし、実際の実装では、異なる引数のタイプに対応する為に、一つの関数を使用
します。この関数は通常、'@overload'デコレータで指定された型のすべての組み合わせ
を処理できるように設計されている。

この例では'process'関数は文字列と整数の両方を引数として受け取ることが出来る。
文字列が渡されると、その文字列を大文字に変換して返す。整数が渡されると、
その整数を2倍にして返す。これはPythonにおけるオーバーロードの一例である。

ただし、'@overload'デコレータは実際の実行時の振る舞いを変更するものではなく、
主に静的型チェッカーやIDEが関数の使用方法を理解するためのものである。
そのため、上記のコードは実際には'@overload'デコレータなしでも同じように動作する。
"""

from typing import overload, Union

# 関数のオーバーロードを定義する
@overload
def process(data: str) -> str:
    ...
    # Ellipsis
    # @overloadデコレータと一緒に使われるとき、実装が後で提供さえることを示す。
    # これはPythonの型ヒントシステムの一部で、静的型チェッカーやIDEが関数の
    # 使用方法を理解するためのもの。実際の実行時には'...'を含む'@overload'
    # デコレータ付きの関数定義は無視される

@overload
def process(data: int) -> int:
    ...

# 実際の実装
def process(data: Union[str, int]) -> Union[str, int]:
    if isinstance(data, str):
        print("Processing strintg data")
        return data.upper()
    elif isinstance(data, int):
        print("Processing integer data")
        return data * 2
    else:
        raise TypeError("Unsupported data type")

# テスト
print(process("hello")) # Processing string data, HELLO
print(process(10)) # Processing integer data, 20

