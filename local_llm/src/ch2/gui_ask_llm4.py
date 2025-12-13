#!/usr/bin/env python

import ollama
import TkEasyGUI as eg

# OllamaのAPIを使うための設定
client = ollama.Client()
client_model = "llama3.2"
# client_model = "phi4"
default_prompt = (
    "親しい友人に手紙を書きます。気の利いた出だしの挨拶を一つ考えてください。"
)

client_result = ""  # モデルの結果を格納する変数

# カスタムレイアウトのウィンドウを作成する
layout = [
    [eg.Text("プロンプトを入力してください")],
    [eg.Multiline(default_prompt, key="-prompt-", size=(60, 3))],
    [eg.Button("実行")],
    [eg.Multiline("", key="-result-", size=(60, 10))],
]
window = eg.Window("LLMに質問する", layout)


# マルチスレッドでLLMに質問する関数を定義
def thread_llm_stream(prompt):
    client_result = ""
    # ストリームモードでLLMに質問して結果を表示する
    stream = client.generate(model=client_model, stream=True, prompt=prompt)
    # 順次ストリームイベントを送出
    for chunk in stream:
        res = chunk["response"]
        window.post_event("ストリーム", {"result": res})
    # イベントをポストしてウィンドウを更新する
    window.post_event("実行完了", {})


# ウィンドウのイベントを処理する
while True:
    event, values = window.read()  # イベントと値を取得する
    if event == eg.WIN_CLOSED:
        break
    if event == "実行":
        # 入力されたプロンプトを取得
        prompt = values["-prompt-"]
        # ボタンを押せないように変更
        window["実行"].update(disables=True)
        # LLMに質問して結果を表示する
        window.start_thread(thread_llm_stream, prompt=prompt)
    elif event == "ストリーム":
        # ストリームイベントを受け取った時
        result = values["result"]
        client_result += result
        # 結果を表示
        window["-result-"].update(client_result)
    elif event == "実行完了":
        # ボタンを押せるように変更
        window["実行"].update(disables=False)
