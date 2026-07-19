"""任意のコマンドをサブプロセスで起動し、stdout から f32le PCM を
読み取る基底クラスです。

FFmpegInput や今後の Rust 製レシーバーなど、stdin/stdout パイプで
通信する外部プロセス全般の共通ロジックを提供します。
"""

import os
import subprocess
import threading
import numpy as np
from logging import getLogger
from inputs.Input import Input

logger = getLogger("app")


class ProcessInput(Input):
    """サブプロセスを起動し、その stdout から f32le PCM を読み取る基底クラス。

    サブクラスは __init__ で self._cmd を設定するだけで、
    プロセス管理・パイプ読み取り・stderr ログは自動的に行われる。
    """

    def __init__(self, cmd: list[str], sample_rate: int, chunk_size: int = 1024):
        self._cmd = cmd
        self._sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._proc: subprocess.Popen | None = None
        self._closed = False
        self._buffer = b""
        self._stderr_thread: threading.Thread | None = None
        self._stderr_stop = threading.Event()

    def _ensure_open(self):
        """サブプロセスを起動します。既に起動済みの場合は何もしません。"""
        if self._proc is not None:
            return

        logger.debug(f"ProcessInput starting: {' '.join(self._cmd)}")
        self._proc = subprocess.Popen(
            self._cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_stop.clear()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        logger.debug("ProcessInput started")

    def _read_stderr(self):
        """stderr を非同期で読み取り、ログに出力します。"""
        assert self._proc is not None and self._proc.stderr is not None
        try:
            for raw_line in self._proc.stderr:
                if self._stderr_stop.is_set():
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    logger.debug(f"[process] {line}")
        except ValueError:
            pass

    def sample_rate(self) -> int:
        """入力音声のサンプルレートを返します。"""
        return self._sample_rate

    def chunks(self, chunk_size: int):
        """与えられたサイズの音声チャンクを yield で返します。"""
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
            logger.error(f"Error reading from process: {e}")
            raise
        finally:
            if self._buffer:
                if len(self._buffer) < bytes_per_chunk:
                    self._buffer += b"\x00" * (bytes_per_chunk - len(self._buffer))
                yield np.frombuffer(self._buffer[:bytes_per_chunk], dtype=np.float32)

    def close(self) -> None:
        """サブプロセスを終了し、リソースを解放します。"""
        self._closed = True
        if self._proc is not None:
            self._stderr_stop.set()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2)
                self._stderr_thread = None
            if self._proc.stdout:
                self._proc.stdout.close()
            if self._proc.stderr:
                self._proc.stderr.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        self._buffer = b""
        logger.debug("ProcessInput closed")
