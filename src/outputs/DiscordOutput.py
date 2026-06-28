import asyncio
import queue
import threading
import numpy as np
import discord
from outputs.Output import Output
from logging import getLogger

logger = getLogger("app")


class _QueueAudioSource(discord.AudioSource):
    """キューから PCM データを読み出すカスタム AudioSource です。"""

    def __init__(self, data_queue: queue.Queue[bytes]) -> None:
        super().__init__()
        self._buffer = b""
        self._queue = data_queue

    def read(self) -> bytes:
        # 3840 bytes = 20ms of s16le stereo 48kHz
        if len(self._buffer) < 3840:
            try:
                chunk = self._queue.get(timeout=0.1)
                self._buffer += chunk
            except queue.Empty:
                return b"\x00" * 3840
        if len(self._buffer) >= 3840:
            data = self._buffer[:3840]
            self._buffer = self._buffer[3840:]
            return data
        return b"\x00" * 3840


class DiscordOutput(Output):
    """Discord ボイスチャンネルに音声を出力するシンクです。"""

    def __init__(self, token: str, channel_id: int, sample_rate: int) -> None:
        self._token = token
        self._channel_id = channel_id
        self._sample_rate = sample_rate
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._ready = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: discord.Client | None = None
        self._voice_client: discord.VoiceClient | None = None

        # 別スレッドで非同期ループを実行
        self._thread = threading.Thread(target=self._run_async, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise TimeoutError("Failed to connect to Discord voice channel")

    def _run_async(self) -> None:
        """非同期メインループを実行します。"""
        asyncio.run(self._async_run())

    async def _async_run(self) -> None:
        """非同期の初期化と Bot の起動を行います。"""
        self._loop = asyncio.get_running_loop()
        intents = discord.Intents.default()
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            """Bot の準備ができたときに VC に参加し、再生を開始します。"""
            try:
                channel = self._client.get_channel(self._channel_id)
                if channel is None:
                    logger.error(f"Channel {self._channel_id} not found")
                    return
                if not isinstance(channel, discord.VoiceChannel):
                    logger.error(f"Channel {self._channel_id} is not a voice channel")
                    return
                self._voice_client = await channel.connect()
                source = _QueueAudioSource(self._queue)
                self._voice_client.play(source)
                logger.info(f"Connected to voice channel: {channel.name}")
            except Exception as e:
                logger.error(f"Failed to connect to voice channel: {e}")
            finally:
                self._ready.set()

        try:
            await self._client.start(self._token)
        finally:
            await self._client.close()

    def write(self, chunk: np.ndarray) -> None:
        """float32 配列を int16 に変換し、キューに投入します。"""
        if self._closed:
            return
        # float32 → int16 に変換
        mono_int16 = (chunk * 32767.0).astype(np.int16)
        # モノラルをステレオ (インターリーブ) に拡張
        stereo = np.repeat(mono_int16, 2)
        self._queue.put(stereo.tobytes())

    def sample_rate(self) -> int:
        """Discord出力のサンプルレートを返します。"""
        return self._sample_rate

    def close(self) -> None:
        """リソースを解放します。"""
        if self._closed:
            return
        self._closed = True
        if self._loop is not None and self._client is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._async_close(), self._loop
            )
            try:
                future.result(timeout=30)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=10)

    async def _async_close(self) -> None:
        """非同期の後片付けを行います。"""
        if self._voice_client is not None:
            try:
                self._voice_client.stop()
                await self._voice_client.disconnect()
            except BaseException:
                pass
        try:
            await self._client.close()
        except BaseException:
            pass
