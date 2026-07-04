<div align="center">
  <h1>VCClient-CLI</h1>
  Supported with 💛 by Kisaragi Project
</div>


## 概要

VCClient-CLI は [VCClient (w-okada/voice-changer)](https://github.com/w-okada/voice-changer) をバックエンドとして使用する RVC 音声変換 CLI ツールです。

## 特徴

CLI であるため、外部から SSH で RVC を実行することができます。

入出力ではローカルファイル・ネットワークを自由にルーティングできます。
このルーティングにはシステムの FFmpeg を利用することもでき、これにより SRT や RTMP などネットワークから直接音声を入力して音声変換を実行できます。

入出力は以下に対応しています:
- WAV ファイル
- オーディオデバイス
- FFmpeg
    - SRT, RTMP, HTTP などのネットワークプロトコル
- Discord Bot

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

## 実行手順

ヘルプを表示:

```sh
python src/main.py
```

つくよみちゃん RVC (モデル ID 0) を使用して、GPU 0 でローカルファイル `awesome_input_audio.wav` を変換し `converted_awesome_audio.wav` に保存する例:

```sh
python src/main.py \
    --model 0 \
    --gpu 0 \
    --input-file awesome_input_audio.wav \
    --output-file converted_awesome_audio.wav
```

Tune 6 (ピッチを3音上げる) に設定し、SRT プロトコルを使用してローカルファイル `awesome_input_audio.wav` を `srt://localhost:1234` に Caller モードとして出力する例:

```sh
python src/main.py \
    --model 0 \
    --tune 6 \
    --gpu 0 \
    --input-type ffmpeg \
    --input-url srt://localhost:1234?mode=caller \
    --input-sample-rate 44100 \
    --output-type ffmpeg \
    --output-url srt://localhost:1234?mode=caller
```

接続されているマイクから音声を取得し、Discord のボイスチャンネルに出力する例:

```sh
python src/main.py \
    --model 0 \
    --gpu 0 \
    --input-type device \
    --input-device "Microphone (Realtek High Definition Audio)" \
    --output-type discord \
    --discord-token DISCORD_TOKEN_HERE \
    --discord-channel DISCORD_VOICE_CHANNEL_ID_HERE
```

### サンプルモデルのダウンロード

VCClient 付属の初期サンプルモデルをダウンロードしたい場合、`--enable-downloading-models` フラグを使用してください。

このフラグを付けていても、すでにサンプルモデルが存在する場合はダウンロードされません。

```sh
python src/main.py --enable-downloading-models
```

## トラブルシューティング

> [!WARNING]
> このツールに関する問題は、VCClient のリポジトリではなく、こちらのリポジトリで報告してください。

### Caller モードで SRT を使用した際の I/O エラー

SRT プロトコルを利用しており、Caller モードで動作している際、一度接続先から切断された後再接続した場合にもこのエラーが出ることがあります。その場合は Listener を再起動してください。

Listener モードに設定した OBS でこの問題が発生する場合、「非アクティブ時にファイルを閉じる」のチェックを無効化してみてください。

### FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.

```
src\vcclient\server\voice_changer\RVC\pipeline\Pipeline.py:120: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast(enabled=self.isHalf):
```

この警告は VCClient のコード内で発生しているものであり、VCClient-CLI で修正する予定はありません。無視しても問題ありません。


### 動作確認済み環境

- Windows 11, Python 3.10.19, GTX1650 (CUDA 12.1)
