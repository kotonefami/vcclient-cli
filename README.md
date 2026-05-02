# VCClient-CLI

[VCClient (w-okada/voice-changer)](https://github.com/w-okada/voice-changer) をベースに、オーディオデバイス以外のソースから音声を入力・出力して RVC 音声変換を実行できる CLI ツールです。

> [!WARNING]
> このツールに関する問題は、VCClient のリポジトリではなく、こちらのリポジトリで報告してください。

## セットアップ手順

Python 3.10.x が必要です。

```sh
# w-okada/voice-changer をサブモジュールとしてクローン
git submodule update --init --depth 1 --single-branch voice-changer

# 依存関係をインストール
pip install -e .
# CUDA 12.1 で GPU 推論する場合
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
```

## 実行方法

```sh
python src/main.py
```

### サンプルモデルのダウンロード

VCClient 付属の初期サンプルモデルをダウンロードしたい場合、`--enable-downloading-models` フラグを使用してください。

このフラグを付けていても、すでにサンプルモデルが存在する場合はダウンロードされません。

```sh
python src/main.py --enable-downloading-models
```

## 動作確認環境

- Windows 11, Python 3.10.19, GTX1650 (CUDA 12.1)
