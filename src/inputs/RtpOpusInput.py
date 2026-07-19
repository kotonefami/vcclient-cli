"""RTP/Opus 受信のための FFmpegInput ラッパークラスです。

固定 SDP ファイルを動的に生成し、FFmpeg の RTP demuxer で受信します。
SDP によりコーデック情報が帯域外通知されるため、
受信側の起動タイミングや途中参加に依存せず安定して Opus ストリームをデコードできます。
"""

import os
import tempfile
from inputs.FFmpegInput import FFmpegInput
from logging import getLogger

logger = getLogger("app")

# channels 1 (mono)
SDP_TEMPLATE = """v=0
o=- 0 0 IN IP4 127.0.0.1
s=RTP Opus Stream
c=IN IP4 0.0.0.0
t=0 0
m=audio {port} RTP/AVP {payload_type}
a=rtpmap:{payload_type} opus/{sample_rate}/1
"""


class RtpOpusInput(FFmpegInput):
    """低遅延 RTP/Opus 通信のためのクラスです。

    FFmpegInput を継承し、RTP/Opus の入力に特化したオプションと
    固定 SDP ファイルの動的生成を注入します。
    """

    def __init__(self, port: int = 20012, payload_type: int = 120, **kwargs):
        self._sdp_path: str | None = None
        sample_rate = kwargs.get("sample_rate", 48000)

        self._sdp_path = self._write_sdp(port, sample_rate, payload_type)
        logger.debug(f"RTP SDP written: {self._sdp_path} (port={port}, sr={sample_rate}, pt={payload_type})")

        input_args = [
            "-protocol_whitelist", "file,rtp,udp",
            "-analyzeduration", "0",
            "-probesize", "32", # TODO: 頭のパケットを落としやすくなるが、RTP/Opus の場合は問題ないか？要検証
            "-flags", "low_delay",
            "-fflags", "nobuffer",
            "-flush_packets", "1",
            "-max_delay", "50000", # TODO: ここを制御できるようにする？
            "-rtbufsize", "256k",
            "-thread_queue_size", "64",
        ]

        super().__init__(url=self._sdp_path, input_args=input_args, **kwargs)

    def _write_sdp(self, port: int, sample_rate: int, payload_type: int) -> str:
        """固定 SDP ファイルを生成し、ファイルパスを返します。"""
        fd, path = tempfile.mkstemp(suffix=".sdp", prefix="rtp_opus_")
        with os.fdopen(fd, "w") as f:
            f.write(SDP_TEMPLATE.format(port=port, sample_rate=sample_rate, payload_type=payload_type))
        return path

    def close(self) -> None:
        """リソースを解放し、一時 SDP ファイルを削除します。"""
        super().close()
        if self._sdp_path is not None:
            try:
                os.unlink(self._sdp_path)
            except OSError:
                pass
            self._sdp_path = None
