import os
import subprocess
import numpy as np
from inputs.Input import Input
import logging

logger = logging.getLogger("app")

class FFmpegInput(Input):
    """FFmpeg を使った汎用入力クラス。

    input_args で入力側のオプションを外部から注入できる。
    指定しない場合は低遅延向けのデフォルト値が使用される。
    """

    DEFAULT_INPUT_ARGS: list[str] = [
        "-analyzeduration", "0",
        "-probesize", "32",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-flush_packets", "1",
    ]

    def __init__(
        self,
        url: str,
        sample_rate: int,
        chunk_size: int = 1024,
        input_args: list[str] | None = None,
    ):
        self.url = url
        self._sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._input_args = input_args if input_args is not None else self.DEFAULT_INPUT_ARGS
        self._proc = None
        self._closed = False
        self._buffer = b""

    def _ensure_open(self):
        if self._proc is not None:
            return

        cmd = [
            "ffmpeg",
            *self._input_args,
            "-i", self.url,
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(self._sample_rate),
            "pipe:1"
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )
        logger.debug("FFmpeg input process started")

    def sample_rate(self) -> int:
        return self._sample_rate

    def chunks(self, chunk_size: int):
        if self._closed:
            return
        self._ensure_open()
        chunk_size = chunk_size or self.chunk_size
        bytes_per_chunk = chunk_size * 4

        try:
            while True:
                data = os.read(self._proc.stdout.fileno(), bytes_per_chunk - len(self._buffer))

                if not data:
                    break
                self._buffer += data
                while len(self._buffer) >= bytes_per_chunk:
                    chunk_data = self._buffer[:bytes_per_chunk]
                    self._buffer = self._buffer[bytes_per_chunk:]
                    yield np.frombuffer(chunk_data, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error reading from ffmpeg: {e}")
            raise
        finally:
            if self._buffer:
                if len(self._buffer) < bytes_per_chunk:
                    self._buffer += b"\x00" * (bytes_per_chunk - len(self._buffer))
                yield np.frombuffer(self._buffer[:bytes_per_chunk], dtype=np.float32)

    def close(self) -> None:
        self._closed = True
        if self._proc is not None:
            if self._proc.stdout:
                self._proc.stdout.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        self._buffer = b""
