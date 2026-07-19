"""FFmpeg を使った汎用出力クラスです。

ProcessOutput を継承し、FFmpeg に特化したコマンド構築を提供します。
"""

import logging
from outputs.ProcessOutput import ProcessOutput

logger = logging.getLogger("app")


class FFmpegOutput(ProcessOutput):
    """FFmpeg を使って音声出力を送信するクラス。"""

    def __init__(self, url: str, sample_rate: int = 44100, codec_name: str = "aac"):
        self.url = url
        self.codec_name = codec_name

        cmd = [
            "ffmpeg",
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-i", "pipe:0",
            "-c:a", self.codec_name,
            "-f", "mpegts",
            "-muxdelay", "0.001",
            "-flush_packets", "1",
            "-pkt_size", "1316",
            self.url,
        ]
        super().__init__(cmd, sample_rate)
