#!/bin/bash

# python 3.11をインストール
uv python install 3.11

mkdir -p ~/open-webui
cd ~/open/webui

# python 3.11を使用する仮想環境を作成する
uv venv --python 3.11

# open-webuiをインストールする
uv pip install open-webui

# サーバーを起動する
# uv runを使うと、仮想環境内でコマンドを実行できる
uv run open-webui server --port 7777
