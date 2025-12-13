#!/usr/bin/env python

import ollama

client = ollama.Client()
client_model = "llama3.2"
prompt = "親しい友人に愛情あふれた手紙を書いてください。"

stream = client.generate(model=client_model, stream=True, prompt=prompt)

# 順次結果を画面に表示する
for chunk in stream:
    subtext = chunk["response"]
    print(subtext, end="", flush=True)
print()
