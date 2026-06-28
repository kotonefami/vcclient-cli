from logging import getLogger
import time
import numpy as np
import sounddevice as sd
from outputs.Output import Output

logger = getLogger("app")


class DeviceOutput(Output):
    """サウンドデバイスに音声を出力するクラスです。"""

    def __init__(self, device: int | str | None, sample_rate: int | None = None) -> None:
        """
        コンストラクタです。

        Args:
            device: 使用するデバイスのインデックス、名前、または None（デフォルト）
            sample_rate: サンプルレート。None の場合はデバイスのデフォルトレートを自動検出します。
        """
        self._device = device
        self._closed = False
        self._write_count = 0
        extra_settings = None

        if sample_rate is not None:
            self._sample_rate = sample_rate
        else:
            try:
                device_info = sd.query_devices(device)
                self._sample_rate = int(device_info["default_samplerate"])
                logger.info(
                    f"Output device sample rate auto-detected: "
                    f"{self._sample_rate} Hz (device={device})"
                )
            except Exception as e:
                self._sample_rate = 48000
                logger.warning(
                    f"Failed to query output device sample rate: {e}. "
                    f"Falling back to {self._sample_rate} Hz."
                )

        # WASAPI 共有モードでは auto_convert=True が必要（PortAudio の二重バッファ回避）
        try:
            device_info = sd.query_devices(device)
            hostapi_info = sd.query_hostapis(device_info["hostapi"])
            if "WASAPI" in hostapi_info["name"]:
                extra_settings = sd.WasapiSettings(auto_convert=True)
                logger.info(
                    f"WASAPI auto_convert enabled for output device "
                    f"(host_api={hostapi_info['name']})"
                )
        except Exception:
            pass

        self._stream: sd.OutputStream | None = sd.OutputStream(
            device=self._device,
            samplerate=self._sample_rate,
            dtype="float32",
            channels=1,
            latency="low",
            extra_settings=extra_settings,
        )
        self._stream.start()

        # ストリーム情報をログ出力（デバッグ用）
        logger.debug(
            f"OutputStream started: device={self._device}, "
            f"samplerate={self._sample_rate}Hz, "
            f"latency={self._stream.latency}"
        )

    def write(self, chunk: np.ndarray) -> None:
        """指定された音声チャンクをデバイスに出力します。"""
        if not self._closed and self._stream is not None:
            start_time = time.perf_counter()
            self._stream.write(chunk.reshape(-1, 1))
            elapsed = time.perf_counter() - start_time

            # 計時情報をログ出力（50回ごと）
            if self._write_count % 50 == 0:
                logger.debug(
                    f"Output write #{self._write_count}: "
                    f"elapsed={elapsed:.4f}s, chunk_shape={chunk.shape}"
                )

            self._write_count += 1

    def sample_rate(self) -> int:
        """出力デバイスのサンプルレートを返します。"""
        return self._sample_rate

    def close(self) -> None:
        """出力リソースを解放し、後処理を行います。"""
        self._closed = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
