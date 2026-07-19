"""任意のコマンドをサブプロセスで起動し、stdin 経由で f32le PCM を
送信する基底クラスです。

FFmpegOutput や今後の Rust 製エンコーダーなど、stdin パイプで
通信する外部プロセス全般の共通ロジックを提供します。
"""

import subprocess
import threading
import numpy as np
from logging import getLogger
from outputs.Output import Output

logger = getLogger("app")


class ProcessOutput(Output):
    """サブプロセスを起動し、その stdin に f32le PCM を書き込む基底クラス。

    サブクラスは __init__ で self._cmd を設定するだけで、
    プロセス管理・stdin 書き込みは自動的に行われる。
    """

    def __init__(self, cmd: list[str], sample_rate: int):
        self._cmd = cmd
        self._sample_rate = sample_rate
        self._proc: subprocess.Popen | None = None
        self._closed = False
        self._stderr_thread: threading.Thread | None = None
        self._stderr_stop = threading.Event()

    def _ensure_open(self):
        """サブプロセスを起動します。既に起動済みの場合は何もしません。"""
        if self._proc is not None:
            return

        logger.debug(f"ProcessOutput starting: {' '.join(self._cmd)}")
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_stop.clear()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        logger.debug("ProcessOutput started")

    def _read_stderr(self):
        """stderr を非同期で読み取り、ログに出力します。"""
        assert self._proc is not None and self._proc.stderr is not None
        try:
            for raw_line in self._proc.stderr:
                if self._stderr_stop.is_set():
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    logger.debug(f"[process-out] {line}")
        except ValueError:
            pass

    def write(self, chunk: np.ndarray) -> None:
        """指定された float32 チャンクをサブプロセスの stdin に書き込みます。"""
        if self._closed:
            return
        self._ensure_open()

        if self._proc.poll() is not None:
            logger.error(f"Process exited with code {self._proc.returncode} before write")
            return

        data = chunk.astype(np.float32).tobytes()
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except Exception as e:
            logger.error(f"Error writing to process: {e}")
            raise

    def sample_rate(self) -> int:
        """出力音声のサンプルレートを返します。"""
        return self._sample_rate

    def close(self) -> None:
        """サブプロセスを終了し、リソースを解放します。"""
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            self._stderr_stop.set()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2)
                self._stderr_thread = None
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            if self._proc.stderr:
                try:
                    self._proc.stderr.close()
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        logger.debug("ProcessOutput closed")
