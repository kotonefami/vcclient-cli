import numpy as np
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

def get_print_volume(audio: np.ndarray):
    """オーディオの RMS を計算し、音量表示を返します。"""
    rms = np.sqrt(np.mean(audio ** 2))
    db = 20 * np.log10(rms + 1e-9)
    rms_display = int(rms * 100)
    meter = "█" * int(rms * 10)
    return f"Input: [{meter:<10}] {rms_display:>3}% ({db:>6.2f} dB)"

def debug_log(data: np.ndarray, is_output: bool = False):
    """
    デバッグ用のログ出力を行います。
    """
    if data.dtype.kind in "iu":
        max_value = float(np.iinfo(data.dtype).max + 1)
    else:
        max_value = 1.0

    # データの統計情報を計算
    peak_val = np.max(np.abs(data)) / max_value
    rms_val = np.sqrt(np.clip(np.mean(data ** 2), a_min=0, a_max=None)) / max_value
    mean_val = data.mean() / max_value

    logger.debug(
        f"{'OUT' if is_output else 'IN '}({str(data.dtype):<7}[{str(len(data)):<5}]): "
        f"Peak={peak_val:.3f}, "
        f"RMS={rms_val:.3f}, "
        f"Mean={mean_val: .3f}"
    )

    # 音割れ（1.0以上でクリッピング）
    if peak_val >= 1.0:
        logger.warning(f"{'OUT' if is_output else 'IN '} data is clipping (peak: {peak_val:.3f}).")

def _parse_ffmpeg_args(raw: str | None) -> list[str] | None:
    """スペース区切りの FFmpeg オプション文字列をリストに変換します。"""
    if raw is None:
        return None
    return raw.split()


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

        input_args = _parse_ffmpeg_args(kwargs.get("input_ffmpeg_args"))

        from inputs.FFmpegInput import FFmpegInput
        return FFmpegInput(input_url, sample_rate, input_args=input_args)
    elif input_type == "ogg_opus":
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("If input type is 'ogg_opus', --input-sample-rate must be specified")
        port = kwargs.get("input_port", 20012)

        from inputs.OggOpusInput import OggOpusInput
        return OggOpusInput(port=port, sample_rate=sample_rate)
    elif input_type == "device":
        device = kwargs.get("input_device")
        sample_rate = kwargs.get("input_device_sample_rate")

        from inputs.DeviceInput import DeviceInput
        if isinstance(device, str) and device.strip().isdigit():
            device = int(device)
        return DeviceInput(device, sample_rate)
    elif input_type == "discord":
        discord_token = kwargs.get("discord_token")
        discord_channel = kwargs.get("discord_channel")
        if discord_token is None:
            raise ValueError("If input type is 'discord', --discord-token must be specified")
        if discord_channel is None:
            raise ValueError("If input type is 'discord', --discord-channel must be specified")

        from inputs.DiscordInput import DiscordInput
        return DiscordInput(discord_token, discord_channel)
    elif input_type == "process":
        cmd = kwargs.get("input_cmd")
        sample_rate = kwargs.get("sample_rate")
        if cmd is None:
            raise ValueError("If input type is 'process', --input-cmd must be specified")
        if sample_rate is None:
            raise ValueError("If input type is 'process', --input-sample-rate must be specified")

        from inputs.ProcessInput import ProcessInput
        return ProcessInput(cmd, sample_rate)
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
    elif output_type == "device":
        device = kwargs.get("output_device")
        sample_rate = kwargs.get("output_device_sample_rate")

        from outputs.DeviceOutput import DeviceOutput
        if isinstance(device, str) and device.strip().isdigit():
            device = int(device)
        return DeviceOutput(device, sample_rate)
    elif output_type == "discord":
        discord_token = kwargs.get("discord_token")
        discord_channel = kwargs.get("discord_channel")
        sample_rate = kwargs.get("sample_rate")
        if discord_token is None:
            raise ValueError("If output type is 'discord', --discord-token must be specified")
        if discord_channel is None:
            raise ValueError("If output type is 'discord', --discord-channel must be specified")
        if sample_rate is None:
            raise ValueError("Output sample rate is not specified")

        from outputs.DiscordOutput import DiscordOutput
        return DiscordOutput(discord_token, discord_channel, sample_rate)
    elif output_type == "process":
        cmd = kwargs.get("output_cmd")
        sample_rate = kwargs.get("sample_rate")
        if cmd is None:
            raise ValueError("If output type is 'process', --output-cmd must be specified")
        if sample_rate is None:
            raise ValueError("Output sample rate is not specified")

        from outputs.ProcessOutput import ProcessOutput
        return ProcessOutput(cmd, sample_rate)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")
