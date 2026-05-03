import subprocess
import numpy as np
from outputs.Output import Output
import logging

logger = logging.getLogger("app")

class FFmpegOutput(Output):
    def __init__(self, url: str, sample_rate: int = 44100, codec_name: str = "aac"):
        self.url = url
        self.sample_rate = sample_rate
        self.codec_name = codec_name
        self._proc = None
        self._closed = False

    def _ensure_open(self):
        if self._proc is not None:
            return
        cmd = [
            "ffmpeg",
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-i", "pipe:0",
            "-c:a", self.codec_name,
            "-f", "mpegts",
            "-muxdelay", "0.001",
            "-flush_packets", "1",
            "-pkt_size", "1316",
            self.url
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )
        logger.debug("FFmpeg output process started")

    def write(self, chunk: np.ndarray) -> None:
        if self._closed:
            return
        self._ensure_open()
        data = chunk.astype(np.float32).tobytes()
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except Exception as e:
            logger.error(f"Error writing to ffmpeg: {e}")
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
