# VCClient-CLI

VCClient-CLI は [VCClient (w-okada/voice-changer)](https://github.com/w-okada/voice-changer) をバックエンドとして使用する RVC 音声変換 CLI ツールです。

## 特徴

CLI であるため、外部から SSH で RVC を実行することができます。

入出力ではローカルファイル・ネットワークを自由にルーティングできます。
このルーティングには PyAV (FFmpeg) を利用することもでき、これにより SRT や RTMP などネットワークから直接音声を入力して音声変換を実行できます。

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
    --input-file awesome_input_audio.wav \
    --output-type ffmpeg \
    --output-url srt://localhost:1234?mode=caller
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

### FFmpeg 入出力に失敗する

#### IndexError: tuple index out of range

SRT プロトコルの Listener モードを利用して入力ストリームを受け取る際この問題が起きることがあります。
現在修正中です。

#### ModuleNotFoundError: No module named 'av'

FFmpeg 入出力には PyAV (`av`) が必要です。
以下のコマンドで依存関係をインストールしてください。

```sh
pip install -e .[ffmpeg]
```

#### av.error.OSError: [Errno 5] I/O error

PyAV がパケットの送信に失敗した可能性があります。
接続先が正しいことを確認してください。

SRT プロトコルを利用しており、Caller モードで動作している際、一度接続先から切断された後再接続した場合にもこのエラーが出ることがあります。その場合は Listener を再起動してください。
Listener モードに設定した OBS でこの問題が発生する場合、「非アクティブ時にファイルを閉じる」のチェックを無効化してみてください。

#### av.error.ProtocolNotFoundError: [Errno 1330794744] Protocol not found

SRT プロトコルを利用してこのエラーが出る場合、インストールされた PyAV が libsrt をサポートしていない可能性があります。
PyAV を 14.x にダウングレードしてください。

```sh
pip install av<15
```

> [!INFO]
> OpenSSL (libcrypto) が同梱されている影響で、一部の環境においてプログラム全体が強制終了する可能性のあるバグが確認されているため、PyAV 16.1.0 以降では意図的に libsrt が無効化された FFmpeg を使用するようになっています。
> 詳細は [PyAV #1972](https://github.com/PyAV-Org/PyAV/issues/1972) を参照してください。

### PyAV のインストールに失敗する

#### コンパイラが走る (wheel の使用を強制する)

コンパイラが意図せず走る場合、wheel を使用しないと判断されている可能性があります。

`--only-binary` オプションを指定して、強制的に wheel から探すようにしてください。

```sh
pip install --only-binary :all: av<15
```

#### その他 (既存の FFmpeg を使用する)

既存の FFmpeg を利用することで、問題が解決する可能性があります。

```sh
pip install av<15 --no-binary av
```

> [!WARNING]
> POSIX 環境が必要になる ([PyAV の README.md](https://github.com/PyAV-Org/PyAV/blob/main/README.md#alternative-installation-methods) を参照) ため、Windows ではこの方法を使用できません。

### 動作確認済み環境

- Windows 11, Python 3.10.19, GTX1650 (CUDA 12.1)
