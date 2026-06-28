import asyncio
import queue
import threading
import numpy as np
import discord
from inputs.Input import Input
from logging import getLogger

logger = getLogger("app")


class _InputSink(discord.Sink):
    """Opus フレームを PCM にデコードするカスタムシンクです。"""

    def __init__(self) -> None:
        super().__init__()
        self._pcm_queue: queue.Queue[np.ndarray] = queue.Queue()

    def write(self, data: bytes, user: int) -> None:
        super().write(data, user)
        audio_data = self.voice_data.get(user)
        if audio_data is None or audio_data.decoder is None:
            return
        try:
            # 20ms の Opus フレームを PCM (s16le) にデコード
            pcm_bytes = audio_data.decoder.decode(data, 960)
            # s16le → float32 に変換
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            # ステレオ (インターリーブ) をモノラルに平均化
            samples = samples.reshape(-1, 2).mean(axis=1)
            self._pcm_queue.put(samples)
        except Exception:
            pass


class DiscordInput(Input):
    """Discord ボイスチャンネルから音声を受信する入力ソースです。"""

    def __init__(self, token: str, channel_id: int) -> None:
        self._token = token
        self._channel_id = channel_id
        self._ready = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: discord.Client | None = None
        self._voice_client: discord.VoiceClient | None = None
        self._sink = _InputSink()

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
            """Bot の準備ができたときに VC に参加します。"""
            try:
                channel = self._client.get_channel(self._channel_id)
                if channel is None:
                    logger.error(f"Channel {self._channel_id} not found")
                    return
                if not isinstance(channel, discord.VoiceChannel):
                    logger.error(f"Channel {self._channel_id} is not a voice channel")
                    return
                self._voice_client = await channel.connect()
                self._voice_client.listen(self._sink)
                logger.info(f"Connected to voice channel: {channel.name}")
            except Exception as e:
                logger.error(f"Failed to connect to voice channel: {e}")
            finally:
                self._ready.set()

        try:
            await self._client.start(self._token)
        finally:
            await self._client.close()

    def sample_rate(self) -> int:
        """サンプリングレート (48000) を返します。"""
        return 48000

    def chunks(self, chunk_size: int):
        """キューから音声データを取得し、chunk_size ごとに yield します。"""
        buffer = np.array([], dtype=np.float32)
        while not self._closed or not self._sink._pcm_queue.empty():
            try:
                samples = self._sink._pcm_queue.get(timeout=0.1)
                buffer = np.concatenate([buffer, samples])
                while len(buffer) >= chunk_size:
                    yield buffer[:chunk_size]
                    buffer = buffer[chunk_size:]
            except queue.Empty:
                if self._closed and self._sink._pcm_queue.empty():
                    break
                continue
        if len(buffer) > 0:
            yield buffer

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
                self._voice_client.stop_listening()
                await self._voice_client.disconnect()
            except BaseException:
                pass
        try:
            await self._client.close()
        except BaseException:
            pass
