#!/usr/bin/env python

import ollama
import TkEasyGUI as eg

# OllamaのAPIを使うための設定
client = ollama.Client()
client_model = "llama3.2"

# 一行入力ダイアログを表示
prompt = eg.input(
    "プロンプトを入力してください", default="白い猫の名前を１つ考えてください"
)
if prompt is None:
    quit()

# LLMに質問して結果を表示する
response = client.generate(model=client_model, prompt=prompt)
result = response["response"]

# 答えをメモダイアログに表示する
eg.popup_memo(result, title="LLMの応答")
