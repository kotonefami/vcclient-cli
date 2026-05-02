import av
import numpy as np
from inputs.Input import Input

class FFmpegInput(Input):
    def __init__(self, url: str, chunk_size: int = 1024):
        self.url = url
        self.chunk_size = chunk_size
        self._container = None
        self._resampler = None
        self._sample_rate = None
        self._closed = False
        self._buffer = np.array([], dtype=np.float32)

    def _ensure_open(self):
        if self._container is None:
            self._container = av.open(self.url, options={"timeout": "30000000"})
            audio_streams = [s for s in self._container.streams if s.type == "audio"]
            if not audio_streams:
                raise RuntimeError("Unable to find audio stream in the input")
            self._audio_stream = audio_streams[0]
            self._sample_rate = self._audio_stream.rate
            self._resampler = av.AudioResampler(format="fltp", layout="mono", rate=self._sample_rate)

    def sample_rate(self) -> int:
        self._ensure_open()
        return self._sample_rate

    def chunks(self, chunk_size: int):
        if self._closed:
            return
        self._ensure_open()
        chunk_size = chunk_size or self.chunk_size

        try:
            for frame in self._container.decode(audio=self._audio_stream.index):
                if self._closed:
                    break
                frame = self._resampler.resample(frame)
                if not frame:
                    continue
                arr = frame[0].to_ndarray().flatten().astype(np.float32)
                self._buffer = np.concatenate([self._buffer, arr])

                while len(self._buffer) >= chunk_size:
                    chunk = self._buffer[:chunk_size]
                    self._buffer = self._buffer[chunk_size:]
                    yield chunk
        finally:
            if self._buffer.size > 0:
                chunk = np.pad(self._buffer, (0, chunk_size - len(self._buffer)))
                yield chunk

    def close(self) -> None:
        self._closed = True
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
