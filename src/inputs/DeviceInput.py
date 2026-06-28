from logging import getLogger
import time
import numpy as np
import sounddevice as sd
from inputs.Input import Input

logger = getLogger("app")


class DeviceInput(Input):
    """サウンドデバイスから音声を入力するクラスです。"""

    def __init__(self, device: int | str | None, sample_rate: int | None = None) -> None:
        """
        コンストラクタです。

        Args:
            device: 使用するデバイスのインデックス、名前、または None（デフォルト）
            sample_rate: サンプルレート。None の場合はデバイスのデフォルトレートを自動検出します。
        """
        self._device = device
        self._stream: sd.InputStream | None = None
        self._closed = False
        self._extra_settings = None

        if sample_rate is not None:
            self._sample_rate = sample_rate
        else:
            try:
                device_info = sd.query_devices(device)
                self._sample_rate = int(device_info["default_samplerate"])
                logger.info(
                    f"Input device sample rate auto-detected: "
                    f"{self._sample_rate} Hz (device={device})"
                )
            except Exception as e:
                self._sample_rate = 48000
                logger.warning(
                    f"Failed to query input device sample rate: {e}. "
                    f"Falling back to {self._sample_rate} Hz."
                )

        # WASAPI 共有モードでは auto_convert=True が必要（PortAudio の二重バッファ回避）
        try:
            device_info = sd.query_devices(device)
            hostapi_info = sd.query_hostapis(device_info["hostapi"])
            if "WASAPI" in hostapi_info["name"]:
                self._extra_settings = sd.WasapiSettings(auto_convert=True)
                logger.info(
                    f"WASAPI auto_convert enabled for input device "
                    f"(host_api={hostapi_info['name']})"
                )
        except Exception:
            pass

    def chunks(self, chunk_size: int):
        """
        与えられたチャンクサイズで音声データを取得し、yield で返します。

        Args:
            chunk_size: 1 回の読み取りで取得するサンプル数

        Yields:
            shape (chunk_size,) の float32 ndarray
        """
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._sample_rate,
            dtype="float32",
            channels=1,
            blocksize=chunk_size,
            latency=0.05,
            extra_settings=self._extra_settings,
        )
        self._stream.start()

        # ストリーム情報をログ出力（デバッグ用）
        logger.debug(
            f"InputStream started: device={self._device}, "
            f"samplerate={self._sample_rate}Hz, "
            f"latency={self._stream.latency}"
        )

        read_count = 0
        while not self._closed:
            try:
                start_time = time.perf_counter()
                data, overflow = self._stream.read(chunk_size)
                elapsed = time.perf_counter() - start_time

                # オーバーフローをログ出力
                if overflow:
                    logger.warning(f"Input overflow detected at read #{read_count}")

                # サンプルレート不一致の検出（初回 + 50回ごと）
                expected = chunk_size / self._sample_rate
                ratio = elapsed / expected
                if ratio > 1.5:
                    logger.warning(
                        f"Sample rate mismatch detected: "
                        f"expected ~{self._sample_rate} Hz, "
                        f"actual ~{chunk_size / elapsed:.0f} Hz "
                        f"(ratio={ratio:.2f}x)"
                    )

                # 計時情報をログ出力（50回ごと）
                if read_count % 50 == 0:
                    logger.debug(
                        f"Input read #{read_count}: elapsed={elapsed:.4f}s, "
                        f"overflow={overflow}, "
                        f"chunk_shape={data.shape}"
                    )

                read_count += 1
                yield data.flatten()
            except Exception as e:
                if not self._closed:
                    logger.error(f"デバイスからの読み取り中にエラー: {e}")
                break

    def sample_rate(self) -> int:
        """入力音声のサンプルレートを返します。"""
        return self._sample_rate

    def close(self) -> None:
        """入力が保持するストリームリソースを解放します。"""
        self._closed = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
