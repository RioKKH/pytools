#!/usr/bin/env python

import ollama

# OllamaのAPIを使うための設定
client = ollama.Client()
client_model = "phi4"  # モデル名を指定する


# Ollamaで手軽にphi4を使うための関数を定義する
def generate(prompt, temperature=0.7):
    response = client.generate(
        model=client_model,
        prompt=prompt,
        options={"temperature": temperature, "stream": True},
    )
    return response["response"]


if __name__ == "__main__":
    # プロンプトを指定する
    prompt = """
        次の手順で最強にユニークな猫の名前を考えてください
        1. 10個の候補を列挙
        2. 名前のユニーク度を10段階で評価する
        3. 最もユニークなものを１つ選ぶ
    """
    print(generate(prompt))  # 実行
