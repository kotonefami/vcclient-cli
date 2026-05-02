import sys
import logging
import argparse
from utils import print_device_info, print_model_info, get_input, get_output

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
    parser.add_argument("--input-type", type=str, default="file", choices=["file", "ffmpeg"], help="Input type ('file' or 'ffmpeg')")
    parser.add_argument("--input-file", type=str, default="data/input.wav", help="Input file path (required when --input-type is 'file')")
    parser.add_argument("--input-url", type=str, default=None, help="Input URL (required when --input-type is 'ffmpeg')")
    parser.add_argument("--input-sample-rate", type=int, default=None, help="Input audio sample rate (required when --input-type is 'ffmpeg')")
    parser.add_argument("--output-type", type=str, default="file", choices=["file", "ffmpeg"], help="Output type ('file' or 'ffmpeg')")
    parser.add_argument("--output-file", type=str, default="data/output.wav", help="Output file path (required when --output-type is 'file')")
    parser.add_argument("--output-url", type=str, default=None, help="Output URL (required when --output-type is 'ffmpeg')")
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
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # ライブラリをロード
    logger.debug(f"Loading libraries")
    from DirectRVCWrapper import DirectRVCWrapper

    # 各種情報をログに出力
    print_device_info(args.gpu)
    print_model_info(args.model_dir, args.model)

    # 入力と出力を作成
    logger.debug("Creating input and output")
    input_source = get_input(args.input_type, input_file=args.input_file, input_url=args.input_url, sample_rate=args.input_sample_rate)
    output_sink = get_output(args.output_type, output_file=args.output_file, output_url=args.output_url, sample_rate=args.output_sample_rate)

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
    for chunk in input_source.chunks(args.chunk_size):
        out_chunk = rvc.process_chunk(chunk)
        if out_chunk is not None:
            output_sink.write(out_chunk)
    output_sink.close()
    input_source.close()
    logger.info(f"Stream closed")

    # 後処理
    rvc.close()

if __name__ == "__main__":
    main()
