# 不足しているPortAudioの開発用ライブラリをシステムにインストールする
$ sudo apt install portaudio19-dev
# Dockerイメージをビルドする
$ docker compose build
# 音声認識と音声合成のサーバを起動する
$ docker compose up -d
# メインプログラムを実行する
$ python app.py

# このプログラムはwhisper.cppが応答しないために動作してくれない。
# 動作確認をしたい場合には、whisper.cppが応答しない理由を探る必要がある。
