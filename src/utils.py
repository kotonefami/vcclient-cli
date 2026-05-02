from logging import getLogger
from inputs.Input import Input
from outputs.Output import Output

logger = getLogger("app")

def print_device_info(gpu_id: int):
    """GPU ID を受け取り、使用するデバイスの情報をログに出力します。"""
    if gpu_id >= 0:
        import torch
        if torch.cuda.is_available():
            logger.info(f"Device: {torch.cuda.get_device_name(gpu_id)} (ID {gpu_id}) / CUDA {torch.version.cuda}")
        else:
            logger.info("Device: CPU")
    else:
        logger.info("Device: CPU")

def print_model_info(model_dir: str, model_id: int):
    """モデル ID を受け取り、モデルの情報をログに出力します。"""
    import json
    import os
    params_path = os.path.join(model_dir, str(model_id), "params.json")
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
            model_name = params.get("name", "Unknown")
            logger.info(f"Model: {model_name} (ID {model_id})")
    else:
        logger.info(f"Model: Unknown (ID {model_id})")

def get_input(input_type: str, **kwargs) -> Input:
    """入力タイプとファイルパスから入力ソースを作成します。"""
    if input_type == "file":
        input_file = kwargs.get("input_file")
        if input_file is None:
            raise ValueError("If input type is 'file', --input-file must be specified")

        from inputs.WavFileInput import WavFileInput
        return WavFileInput(input_file)
    elif input_type == "ffmpeg":
        input_url = kwargs.get("input_url")
        if input_url is None:
            raise ValueError("If input type is 'ffmpeg', --input-url must be specified")
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("If input type is 'ffmpeg', --input-sample-rate must be specified")

        from inputs.FFmpegInput import FFmpegInput
        return FFmpegInput(input_url, sample_rate)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

def get_output(output_type: str, **kwargs) -> Output:
    """出力タイプとファイルパスから出力シンクを作成します。"""
    if output_type == "file":
        output_file = kwargs.get("output_file")
        sample_rate = kwargs.get("sample_rate")
        if output_file is None:
            raise ValueError("If output type is 'file', --output-file must be specified")
        if sample_rate is None:
            raise ValueError("Output sample rate is not specified")

        from outputs.WavFileOutput import WavFileOutput
        return WavFileOutput(output_file, sample_rate)
    elif output_type == "ffmpeg":
        output_url = kwargs.get("output_url")
        sample_rate = kwargs.get("sample_rate")
        if output_url is None:
            raise ValueError("If output type is 'ffmpeg', --output-url must be specified")
        if sample_rate is None:
            raise ValueError("Output sample rate is not specified")

        from outputs.FFmpegOutput import FFmpegOutput
        return FFmpegOutput(output_url, sample_rate)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")
