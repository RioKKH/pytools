FROM ubuntu:24.04

# Whisper.cppのビルドに必要なパッケージを追加する
RUN rm -rf /var/lib/apt/lists/* \
  && apt-get clean \
  && apt-get update \
  && DEBIAN_FRONTEND=noninteractive \
     apt-get install -y --no-install-recommends \
     curl git build-essential cmake ca-certificates \
  && apt-get clean \
  && rm -rf /var/lib/apt/lits/*

ENV LANG=C.UTF-8
ENV WORKDIR=/app/whisper.cpp

# Whisper.cppのインストール先を作成する
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR
# GitHubからリポジトリの取得
RUN git clone --branch v1.7.5 \
    https://github.com/ggerganov/whisper.cpp.git $WORKDIR
# ビルド
RUN make
# モデルのダウンロード
RUN make -j small

# Whisper.cppのサーバーを起動する
CMD ["./build/bin/whisper-server", \
     "-m", "./models/ggml-small.bin", \
     "--port", "50022", "--host", "0.0.0.0"]
