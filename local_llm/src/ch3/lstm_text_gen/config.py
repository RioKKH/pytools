#!/usr/bin/env python

import os

# プロジェクトで利用するフォルダーのパスを指定
DIR_ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_CORPUS = os.path.join(DIR_ROOT, "corpus")
DIR_MODEL = os.path.join(DIR_ROOT, "model")

# コーパス用のファイルのパスを指定する
FILE_AOZORA_ZIP = os.path.join(DIR_CORPUS, "aozora.zip")
FILE_CORPUS_TXT = os.path.join(DIR_CORPUS, "corpus.txt")

# データセットと辞書データの保存先ファイルパスを指定する
FILE_TOKEN2ID = os.path.join(DIR_MODEL, "token2id.json")
FILE_ID2TOKEN = os.path.join(DIR_MODEL, "id2token.json")
FILE_IDS = os.path.join(DIR_MODEL, "ids.json")

# テキスト生成モデルの保存先ファイルパスを指定する
FILE_MODEL = os.path.join(DIR_MODEL, "model_pth")
