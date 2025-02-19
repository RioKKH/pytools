#!/usr/bin/env python
# coding: utf-8

import pyxel

# 画面を初期化する
pyxel.init(160, 120, title="Pyxel Drawing")

# 点を描く
pyxel.pset(10, 10, 7)

# 線を描く
pyxel.line(150, 10, 10, 110, 8)

# 画面を表示する
pyxel.show()
