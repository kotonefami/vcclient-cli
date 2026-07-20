import sys
import time
import logging
import shlex
import argparse
import numpy as np
import resampy
import scipy.io.wavfile as wavfile
from utils import print_device_info, print_model_info, get_print_volume, get_input, get_output

# アプリのロガーをセットアップ
logger = logging.getLogger("app")
logger.propagate = False # ルートロガーは server/mods/log_control.py で無効化されてしまう
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

def main():
    # コマンドライン引数を定義、引数がない場合はヘルプを表示する
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="data/models", help="Directory containing the models and index")
    parser.add_argument("--pretrain-dir", type=str, default="data/pretrain", help="Directory containing the pretrain models")
    parser.add_argument("--enable-downloading-models", default=False, action="store_true", help="Automatically download initial models if not present")
    parser.add_argument("--model", type=int, required=True, help="Model number to use")
    parser.add_argument("--discord-token", type=str, default=None, help="Discord bot token")
    parser.add_argument("--discord-channel", type=int, default=None, help="Discord voice channel ID")
    parser.add_argument("--input-type", type=str, default="file", choices=["file", "ffmpeg", "ogg_opus", "discord", "device", "process"], help="Input type ('file', 'ffmpeg', 'ogg_opus', 'discord', 'device', or 'process')")
    parser.add_argument("--input-file", type=str, default="data/input.wav", help="Input file path (required when --input-type is 'file')")
    parser.add_argument("--input-url", type=str, default=None, help="Input URL (required when --input-type is 'ffmpeg')")
    parser.add_argument("--input-sample-rate", type=int, default=None, help="Input audio sample rate (required when --input-type is 'ffmpeg')")
    parser.add_argument("--input-port", type=int, default=20012, help="UDP port for OggOpus input (default: 20012)")
    parser.add_argument("--input-ffmpeg-args", type=str, default=None, help="Custom FFmpeg input options (space-separated, used with 'ffmpeg' type)")
    parser.add_argument("--input-cmd", type=str, default=None, help="Command for 'process' input type (use double quotes for arguments, e.g. 'my_program --arg1 val1')")
    parser.add_argument("--input-device", type=str, default=None, help="Input audio device name or index (required when --input-type is 'device')")
    parser.add_argument("--output-device", type=str, default=None, help="Output audio device name or index (required when --output-type is 'device')")
    parser.add_argument("--device-sample-rate", type=int, default=None, help="[Deprecated] Sample rate for device I/O. Use --input-device-sample-rate and --output-device-sample-rate instead.")
    parser.add_argument("--input-device-sample-rate", type=int, default=None, help="Sample rate for input device (auto-detected from device if not specified)")
    parser.add_argument("--output-device-sample-rate", type=int, default=None, help="Sample rate for output device (auto-detected from device if not specified)")
    parser.add_argument("--output-type", type=str, default="file", choices=["file", "ffmpeg", "discord", "device", "process"], help="Output type ('file', 'ffmpeg', 'discord', 'device', or 'process')")
    parser.add_argument("--output-file", type=str, default="data/output.wav", help="Output file path (required when --output-type is 'file')")
    parser.add_argument("--output-url", type=str, default=None, help="Output URL (required when --output-type is 'ffmpeg')")
    parser.add_argument("--output-cmd", type=str, default=None, help="Command for 'process' output type (use double quotes for arguments, e.g. 'my_program --arg1 val1')")
    parser.add_argument("--output-sample-rate", type=int, default=48000, help="Output audio sample rate (default: 48000)")
    parser.add_argument("--f0-detector", type=str, default="rmvpe_onnx", help="F0 detection method to use", choices=["dio", "harvest", "crepe", "crepe_tiny", "crepe_full", "rmvpe", "rmvpe_onnx", "fcpe"])
    parser.add_argument("--tune", type=int, default=0, help="Pitch shift (semitones)")
    parser.add_argument("--chunk-size", type=int, default=4096 * 4, help="Chunk size for pseudo real-time processing")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID to use (CPU if -1)")
    parser.add_argument("--silent-threshold", type=float, default=0.00001, help="Silent threshold for silence detection")
    parser.add_argument("--extra-convert-size", type=int, default=1024 * 4, help="Extra convert size for RVC processing")
    parser.add_argument("--index-ratio", type=float, default=0, help="Index ratio for RVC model")
    parser.add_argument("--protect", type=float, default=0.5, help="Protect setting for RVC")
    parser.add_argument("--rvc-quality", type=int, default=0, help="RVC quality setting")
    parser.add_argument("--silence-front", type=int, default=1, help="Silence front setting (0: off, 1: on)")
    parser.add_argument("--bypass", default=False, action="store_true", help="Bypass RVC processing (passthrough mode)")
    parser.add_argument("--debug-save-input", type=str, default=None, help="Save input audio to WAV file for debugging")
    parser.add_argument("--debug-save-output", type=str, default=None, help="Save output audio to WAV file for debugging")
    parser.add_argument("--performance", default=False, action="store_true", help="Enable performance mode")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # Discord 関連のバリデーション
    if args.input_type == "discord" or args.output_type == "discord":
        if args.discord_token is None:
            parser.error("--discord-token is required when --input-type or --output-type is 'discord'")
        if args.discord_channel is None:
            parser.error("--discord-channel is required when --input-type or --output-type is 'discord'")
    if args.input_type == "discord" and args.output_type == "discord":
        parser.error("--input-type and --output-type cannot both be 'discord'")

    # process タイプのバリデーション
    if args.input_type == "process":
        if args.input_cmd is None:
            parser.error("--input-cmd is required when --input-type is 'process'")
    if args.output_type == "process":
        if args.output_cmd is None:
            parser.error("--output-cmd is required when --output-type is 'process'")

    # --device-sample-rate（非推奨）から新しい引数へのフォールバック
    if args.device_sample_rate is not None:
        logger.warning(
            "--device-sample-rate は非推奨です。"
            "--input-device-sample-rate と --output-device-sample-rate を使用してください。"
        )
        if args.input_device_sample_rate is None:
            args.input_device_sample_rate = args.device_sample_rate
        if args.output_device_sample_rate is None:
            args.output_device_sample_rate = args.device_sample_rate

    # ライブラリをロード
    logger.debug(f"Loading libraries")
    from DirectRVCWrapper import DirectRVCWrapper

    # 各種情報をログに出力
    print_device_info(args.gpu)
    print_model_info(args.model_dir, args.model)

    # 入力と出力を作成
    logger.debug("Creating input and output")
    input_source = get_input(args.input_type, input_file=args.input_file, input_url=args.input_url, sample_rate=args.input_sample_rate, discord_token=args.discord_token, discord_channel=args.discord_channel, input_device=args.input_device, input_device_sample_rate=args.input_device_sample_rate, input_port=args.input_port, input_ffmpeg_args=args.input_ffmpeg_args, input_cmd=shlex.split(args.input_cmd) if args.input_cmd is not None else None)
    output_sink = get_output(args.output_type, output_file=args.output_file, output_url=args.output_url, sample_rate=args.output_sample_rate, discord_token=args.discord_token, discord_channel=args.discord_channel, output_device=args.output_device, output_device_sample_rate=args.output_device_sample_rate, output_cmd=shlex.split(args.output_cmd) if args.output_cmd is not None else None)

    # 入出力のサンプルレートを取得（バイパス時のリサンプリングに使用）
    _input_sr = input_source.sample_rate()
    _output_sr = output_sink.sample_rate()
    _need_resample = _input_sr != _output_sr
    logger.info(f"Input sample rate: {_input_sr} Hz, Output sample rate: {_output_sr} Hz" + (" (resampling enabled)" if _need_resample else ""))

    # RVC を初期化
    logger.debug("Initializing RVC")
    rvc = DirectRVCWrapper(model_dir=args.model_dir, pretrain_dir=args.pretrain_dir)
    if args.enable_downloading_models:
        rvc.download_initial_models()
    rvc.initialize(
        gpu_id=args.gpu,
        model=args.model,
        input_sample_rate=input_source.sample_rate(),
        output_sample_rate=args.output_sample_rate,
        f0_method=args.f0_detector,
        f0_up_key=args.tune,
        silent_threshold=args.silent_threshold,
        extra_convert_size=args.extra_convert_size,
        index_ratio=args.index_ratio,
        protect=args.protect,
        rvc_quality=args.rvc_quality,
        silence_front=args.silence_front,
    )

    # ストリーム処理
    logger.info("Stream started")

    # デバッグ用のチャンク収集リスト
    _debug_input_chunks: list[np.ndarray] = []
    _debug_output_chunks: list[np.ndarray] = []
    _chunk_count = 0
    _loop_start_time = time.perf_counter()
    _prev_loop_time = _loop_start_time

    try:
        for chunk in input_source.chunks(args.chunk_size):
            # 入力チャンクのデバッグログ（50チャンクごと）
            if _chunk_count % 50 == 0:
                logger.debug(
                    f"Input chunk #{_chunk_count}: "
                    f"shape={chunk.shape}, dtype={chunk.dtype}, "
                    f"min={chunk.min():.6f}, max={chunk.max():.6f}, "
                    f"mean={chunk.mean():.6f}, std={chunk.std():.6f}"
                )

            if args.bypass:
                if _need_resample:
                    out_chunk = resampy.resample(chunk, _input_sr, _output_sr)
                else:
                    out_chunk = chunk.copy()
            else:
                if args.performance:
                    start_time = time.perf_counter()
                out_chunk = rvc.process_chunk(chunk)

            if out_chunk is not None:
                # 出力チャンクのデバッグログ（50チャンクごと）
                if _chunk_count % 50 == 0:
                    logger.debug(
                        f"Output chunk #{_chunk_count}: "
                        f"shape={out_chunk.shape}, dtype={out_chunk.dtype}, "
                        f"min={out_chunk.min():.6f}, max={out_chunk.max():.6f}, "
                        f"mean={out_chunk.mean():.6f}, std={out_chunk.std():.6f}"
                    )

                # デバッグ用 WAV 保存のためのチャンク収集
                if args.debug_save_input is not None:
                    _debug_input_chunks.append(chunk.copy())
                if args.debug_save_output is not None:
                    _debug_output_chunks.append(out_chunk.copy())

                if args.performance and not args.bypass:
                    time_taken = time.perf_counter() - start_time
                    chunk_duration = len(chunk) / input_source.sample_rate()
                    print(f"\r{get_print_volume(chunk)} | Processing time: {time_taken:.4f}s (Chunk duration: {chunk_duration:.4f}s)", end="")

                output_sink.write(out_chunk)

                # メインループの計時情報（50イテレーションごと）
                if _chunk_count % 50 == 0:
                    current_time = time.perf_counter()
                    loop_elapsed = current_time - _prev_loop_time
                    chunk_duration = len(chunk) / input_source.sample_rate()
                    logger.debug(
                        f"Main loop iteration #{_chunk_count}: "
                        f"loop_elapsed={loop_elapsed:.4f}s, "
                        f"chunk_duration={chunk_duration:.4f}s, "
                        f"ratio={loop_elapsed/chunk_duration:.2f}x"
                    )
                    _prev_loop_time = current_time

            _chunk_count += 1
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    finally:
        output_sink.close()
        input_source.close()

    # デバッグ WAV 保存
    if args.debug_save_input is not None and _debug_input_chunks:
        _all_input = np.concatenate(_debug_input_chunks)
        _input_sr = input_source.sample_rate()
        wavfile.write(args.debug_save_input, _input_sr, (_all_input * 32767.0).astype(np.int16))
        logger.info(f"Saved input debug WAV: {args.debug_save_input} ({len(_all_input)} samples, {_input_sr} Hz)")

    if args.debug_save_output is not None and _debug_output_chunks:
        _all_output = np.concatenate(_debug_output_chunks)
        _output_sr = args.output_sample_rate
        wavfile.write(args.debug_save_output, _output_sr, (_all_output * 32767.0).astype(np.int16))
        logger.info(f"Saved output debug WAV: {args.debug_save_output} ({len(_all_output)} samples, {_output_sr} Hz)")

    logger.info(f"Stream closed")

    # 後処理
    rvc.close()

if __name__ == "__main__":
    main()
